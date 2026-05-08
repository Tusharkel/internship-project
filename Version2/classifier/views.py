import csv
import os
from datetime import datetime

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from django.shortcuts import render

from classifier.utils.model_loader import model_loader
from classifier.ml.preprocess import clean_text


def _save_to_csv(record: dict):
    """Append a classified email record to the CSV store."""
    path   = settings.CSV_STORE_PATH
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def _read_csv() -> list:
    """Read all records from the CSV store."""
    path = settings.CSV_STORE_PATH
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Simple text-based classify (uses TF-IDF style text input) ────────
class ClassifyTextView(APIView):
    """
    POST /api/classify/
    {
        "subject": "Win a free iPhone!",
        "body":    "Click here to claim your prize now",
        "sender":  "promo@scam.com"   (optional)
    }
    """

    def post(self, request):
        subject = request.data.get("subject", "")
        body    = request.data.get("body",    "")
        sender  = request.data.get("sender",  "unknown")

        if not body:
            return Response(
                {"error": "body field is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # For text input we use a simple heuristic scoring
        # against known spam keywords (since the Kaggle model
        # uses word-frequency vectors not raw text)
        spam_keywords = [
            "free", "win", "winner", "prize", "claim", "click",
            "offer", "congratulations", "urgent", "act now",
            "limited time", "cash", "credit", "loan", "viagra",
            "cheap", "buy now", "discount", "guarantee", "money back"
        ]
        combined   = f"{subject} {body}".lower()
        hits       = sum(1 for kw in spam_keywords if kw in combined)
        confidence = min(0.5 + hits * 0.08, 0.99)
        label      = "spam" if hits >= 3 else "ham"

        record = {
            "id":         datetime.utcnow().strftime("%Y%m%d%H%M%S%f"),
            "subject":    subject[:200],
            "body":       body[:500],
            "sender":     sender,
            "label":      label,
            "confidence": confidence,
            "flagged":    False,
            "created_at": datetime.utcnow().isoformat(),
        }
        _save_to_csv(record)

        return Response({
            "id":         record["id"],
            "label":      label,
            "is_spam":    label == "spam",
            "confidence": round(confidence, 4),
        }, status=status.HTTP_200_OK)


class EmailListView(APIView):
    """
    GET /api/emails/             → all emails
    GET /api/emails/?label=spam  → filtered
    """

    def get(self, request):
        records = _read_csv()
        label   = request.query_params.get("label")
        if label in ("spam", "ham"):
            records = [r for r in records if r["label"] == label]
        return Response({
            "total":  len(records),
            "emails": records,
        })


class EmailDetailView(APIView):
    """
    GET    /api/emails/<id>/  → single email
    DELETE /api/emails/<id>/  → delete email
    """

    def get(self, request, email_id):
        records = _read_csv()
        match   = next((r for r in records if r["id"] == email_id), None)
        if not match:
            return Response({"error": "Not found"}, status=404)
        return Response(match)

    def delete(self, request, email_id):
        records = _read_csv()
        updated = [r for r in records if r["id"] != email_id]
        if len(updated) == len(records):
            return Response({"error": "Not found"}, status=404)
        path = settings.CSV_STORE_PATH
        if updated:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=updated[0].keys())
                writer.writeheader()
                writer.writerows(updated)
        else:
            os.remove(path)
        return Response({"deleted": email_id})


class FlagEmailView(APIView):
    """
    PATCH /api/emails/<id>/flag/
    { "correct_label": "ham" }
    """

    def patch(self, request, email_id):
        correct = request.data.get("correct_label")
        if correct not in ("spam", "ham"):
            return Response({"error": "correct_label must be spam or ham"}, status=400)
        records = _read_csv()
        updated = False
        for r in records:
            if r["id"] == email_id:
                r["label"]   = correct
                r["flagged"] = True
                updated = True
                break
        if not updated:
            return Response({"error": "Not found"}, status=404)
        path = settings.CSV_STORE_PATH
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        return Response({"message": "Flagged successfully", "id": email_id, "new_label": correct})


class StatsView(APIView):
    """GET /api/stats/"""

    def get(self, request):
        records = _read_csv()
        total   = len(records)
        spam    = sum(1 for r in records if r["label"] == "spam")
        ham     = total - spam
        flagged = sum(1 for r in records if str(r.get("flagged")) == "True")
        return Response({
            "total":     total,
            "spam":      spam,
            "ham":       ham,
            "flagged":   flagged,
            "spam_rate": round(spam / total, 4) if total else 0,
        })