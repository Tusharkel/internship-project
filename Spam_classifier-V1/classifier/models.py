import mongoengine as me
from datetime import datetime


class Email(me.Document):
    """Persisted email document in MongoDB."""
    subject    = me.StringField(default="")
    body       = me.StringField(required=True)
    sender     = me.StringField(default="unknown")
    label      = me.StringField(choices=["spam", "ham"], required=True)
    confidence = me.FloatField(min_value=0.0, max_value=1.0)
    flagged    = me.BooleanField(default=False)   # manual correction flag
    created_at = me.DateTimeField(default=datetime.utcnow)

    meta = {
        "collection": "emails",
        "indexes": ["label", "sender", "created_at"],
        "ordering": ["-created_at"],
    }

    def to_dict(self):
        return {
            "id": str(self.id),
            "subject": self.subject,
            "body": self.body[:200] + "..." if len(self.body) > 200 else self.body,
            "sender": self.sender,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "flagged": self.flagged,
            "created_at": self.created_at.isoformat(),
        }