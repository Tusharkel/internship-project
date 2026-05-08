from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from classifier.models import Email
from classifier.ml.predict import classify_email
from classifier.serializers import ClassifyRequestSerializer, FlagRequestSerializer


class ClassifyEmailView(APIView):
    """
    POST /api/classify/
    Body: { subject, body, sender, save }
    Returns: { label, confidence, email_id (if saved) }
    """

    def post(self, request):
        serializer = ClassifyRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data   = serializer.validated_data
        result = classify_email(data["subject"], data["body"])

        response = {
            "label":      result["label"],
            "confidence": round(result["confidence"], 4),
            "is_spam":    result["label"] == "spam",
        }

        if data["save"]:
            email = Email(
                subject    = data["subject"],
                body       = data["body"],
                sender     = data["sender"],
                label      = result["label"],
                confidence = result["confidence"],
            )
            email.save()
            response["email_id"] = str(email.id)

        return Response(response, status=status.HTTP_200_OK)


class EmailListView(APIView):
    """
    GET /api/emails/?label=spam&limit=20&skip=0
    Returns paginated list of classified emails.
    """

    def get(self, request):
        label  = request.query_params.get("label")
        limit  = int(request.query_params.get("limit", 20))
        skip   = int(request.query_params.get("skip", 0))

        qs = Email.objects
        if label in ("spam", "ham"):
            qs = qs.filter(label=label)

        total  = qs.count()
        emails = qs.skip(skip).limit(limit)

        return Response({
            "total":  total,
            "limit":  limit,
            "skip":   skip,
            "emails": [e.to_dict() for e in emails],
        })


class EmailDetailView(APIView):
    """
    GET  /api/emails/<id>/   → retrieve single email
    DELETE /api/emails/<id>/ → delete email
    """

    def _get_email(self, email_id):
        try:
            return Email.objects.get(id=email_id)
        except Exception:
            return None

    def get(self, request, email_id):
        email = self._get_email(email_id)
        if not email:
            return Response({"error": "Not found"}, status=404)
        return Response(email.to_dict())

    def delete(self, request, email_id):
        email = self._get_email(email_id)
        if not email:
            return Response({"error": "Not found"}, status=404)
        email.delete()
        return Response({"deleted": email_id}, status=200)


class FlagEmailView(APIView):
    """
    PATCH /api/emails/<id>/flag/
    Body: { correct_label: "spam"|"ham" }
    Marks an email as incorrectly classified (for future retraining).
    """

    def patch(self, request, email_id):
        try:
            email = Email.objects.get(id=email_id)
        except Exception:
            return Response({"error": "Not found"}, status=404)

        serializer = FlagRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        email.label   = serializer.validated_data["correct_label"]
        email.flagged = True
        email.save()

        return Response({"message": "Email flagged for retraining", **email.to_dict()})


class StatsView(APIView):
    """
    GET /api/stats/
    Returns basic classification stats.
    """

    def get(self, request):
        total   = Email.objects.count()
        spam    = Email.objects.filter(label="spam").count()
        ham     = Email.objects.filter(label="ham").count()
        flagged = Email.objects.filter(flagged=True).count()

        return Response({
            "total":         total,
            "spam":          spam,
            "ham":           ham,
            "flagged":       flagged,
            "spam_rate":     round(spam / total, 4) if total else 0,
        })