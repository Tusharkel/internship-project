from rest_framework import serializers


class ClassifyRequestSerializer(serializers.Serializer):
    subject = serializers.CharField(required=False, default="", allow_blank=True)
    body    = serializers.CharField(required=True)
    sender  = serializers.CharField(required=False, default="unknown")
    save    = serializers.BooleanField(required=False, default=True)


class FlagRequestSerializer(serializers.Serializer):
    correct_label = serializers.ChoiceField(choices=["spam", "ham"])