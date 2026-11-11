"""The 13 audit expectations are now part of the normal regression suite."""
import json
import unittest

from audit_pipeline import CASES, Fixture


class AuditRegressions(unittest.TestCase):
    pass


for _, title, probe in CASES:
    def check(self, probe=probe, title=title):
        fixture = Fixture()
        fixture.setUp()
        try:
            passed, details = probe(fixture)
            self.assertTrue(passed, title + ": " + json.dumps(details, default=str))
        finally:
            fixture.doCleanups()
    setattr(AuditRegressions, "test_" + probe.__name__, check)
