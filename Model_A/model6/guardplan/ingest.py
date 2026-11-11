from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
from pathlib import Path

from .adapter import DatasetAdapter
from .core import GuardPlanError, MemoryRecord, read_json, write_json
from .privacy import Privacy


class Ingestor:
    def __init__(self, adapter: DatasetAdapter, local_model):
        self.adapter, self.model = adapter, local_model
        self.cache = adapter.workspace / "extractions" / adapter.persona_id

    def extract(self, path: str) -> str:
        raw = self.adapter.read_file(path)
        digest = hashlib.sha256(raw).hexdigest()
        # Version/model cache key ensures changing the VLM invalidates image OCR.
        model = getattr(getattr(self.model, "settings", None), "model", "test")
        settings = getattr(self.model, "settings", None)
        extraction_settings = {key: getattr(settings, key, None) for key in
                               ("quantization", "temperature", "max_tokens", "context_tokens", "seed")}
        key = hashlib.sha256((digest + model + json.dumps(extraction_settings, sort_keys=True) + "extraction-v2").encode()).hexdigest()
        cached = self.cache / (key + ".json")
        if cached.exists():
            return read_json(cached)["text"]
        suffix = Path(path).suffix.lower()
        if suffix in {".md", ".csv", ".txt"}:
            text = raw.decode("utf-8-sig")
        elif suffix == ".pdf":
            from pypdf import PdfReader
            pages = []
            reader = PdfReader(io.BytesIO(raw))
            for index, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if len(text.strip()) < 20:
                    text = self._ocr_pdf_page(raw, index)
                pages.append(f"Page {index + 1}\n{text}")
            text = "\n\n".join(pages)
        elif suffix in {".png", ".jpg", ".jpeg"}:
            with tempfile.TemporaryDirectory(dir=self.adapter.workspace) as temp:
                image = Path(temp) / ("image" + suffix)
                image.write_bytes(raw)
                text = self.model.extract_image(image)
        else:
            raise GuardPlanError(f"Unsupported document extension: {suffix}")
        if not text.strip():
            raise GuardPlanError(f"Empty document extraction: {path}")
        write_json(cached, {"source": path, "sha256": digest, "method": suffix, "text": text})
        return text

    def _ocr_pdf_page(self, raw: bytes, index: int) -> str:
        import pypdfium2
        with tempfile.TemporaryDirectory(dir=self.adapter.workspace) as temp:
            path = Path(temp) / "page.png"
            pdf = pypdfium2.PdfDocument(raw)
            page = pdf[index]
            bitmap = page.render(scale=2)
            bitmap.to_pil().save(path)
            text = self.model.extract_image(path)
            bitmap.close()
            page.close()
            pdf.close()
            return text

    def records(self, include_files: bool = True, state_filters: bool = True) -> list[MemoryRecord]:
        self.adapter.workspace.mkdir(parents=True, exist_ok=True)
        profile = self.adapter.get_profile()
        privacy = Privacy(self.adapter.privacy_policy(), profile)
        records = []
        def add(identifier, source, kind, data, valid_from=None, date=None):
            text = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
            marked = isinstance(data, dict) and data.get("is_sensitive", False)
            records.append(MemoryRecord(
                id=identifier, persona_id=self.adapter.persona_id, source=source, type=kind,
                text=text, date=date, valid_from=valid_from,
                sensitivity=privacy.classify(text, marked), data=data if isinstance(data, dict) else {}))
        for event in self.adapter.search_events():
            add(event["id"], "streams/calendar.json", "calendar_event", event, date=event["date"])
        messages = self.adapter.search_messages() if state_filters else self.adapter.unfiltered_messages_for_ablation()
        for message in messages:
            add(message["id"], "streams/messages.json", "message", message,
                valid_from=message["timestamp"], date=message["timestamp"][:10])
        for contact in profile.get("contacts", []):
            add("contact_" + contact["id"], "agent_memory/user_profile.json", "contact", contact)
        for key, value in profile.get("preferences", {}).items():
            add("preference_" + key, "agent_memory/user_profile.json", "preference", {key: value})
        for i, routine in enumerate(profile.get("routines", [])):
            add(f"routine_{i:03d}", "agent_memory/user_profile.json", "routine", routine)
        # Retain other profile facts, including client ownership for NDA typing.
        for key, value in profile.items():
            if key not in {"contacts", "preferences", "routines", "action_execution_policy"}:
                add("profile_" + key, "agent_memory/user_profile.json", "profile", {key: value})
        if include_files:
            for path in self.adapter.list_files():
                text = self.extract(path)
                if Path(path).suffix == ".csv":
                    chunks = [json.dumps(row) for row in csv.DictReader(io.StringIO(text))]
                else:
                    chunks = self.chunks(text)
                for i, chunk in enumerate(chunks):
                    add(f"file_{Path(path).stem}_{i:03d}", path, "file", chunk)
        write_json(self.adapter.workspace / "memory" / self.adapter.persona_id / "records.json",
                   [r.to_dict() for r in records])
        return records

    @staticmethod
    def chunks(text: str, size: int = 1400, overlap: int = 150) -> list[str]:
        # Prefer section/paragraph boundaries, with overlap for long sections.
        chunks, current = [], ""
        for paragraph in text.split("\n\n"):
            if current and len(current) + len(paragraph) > size:
                chunks.append(current)
                current = ""
            while len(paragraph) > size:
                chunks.append(paragraph[:size])
                paragraph = paragraph[size - overlap:]
            current = (current + "\n\n" + paragraph).strip()
        if current:
            chunks.append(current)
        return chunks
