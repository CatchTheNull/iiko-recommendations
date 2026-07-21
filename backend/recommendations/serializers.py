import datetime

from rest_framework import serializers


class IikoServerConfigSerializer(serializers.Serializer):
    url = serializers.CharField()
    login = serializers.CharField()
    password = serializers.CharField()
    date_from = serializers.DateField(required=False)
    date_to = serializers.DateField(required=False)

    def validate(self, attrs):
        attrs.setdefault("date_from", datetime.date.today() - datetime.timedelta(days=30))
        attrs.setdefault("date_to", datetime.date.today())
        return attrs


class TransportConfigSerializer(serializers.Serializer):
    api_key = serializers.CharField()
    organization_id = serializers.CharField()
    external_menu_id = serializers.CharField()


class RecoSettingsSerializer(serializers.Serializer):
    top_n = serializers.IntegerField(required=False, default=8, min_value=1, max_value=50)
    min_co = serializers.IntegerField(required=False, default=1, min_value=1)
    excluded_categories = serializers.ListField(
        child=serializers.CharField(), required=False, default=list
    )


class RecommendationsRequestSerializer(serializers.Serializer):
    iiko_server = IikoServerConfigSerializer()
    transport = TransportConfigSerializer(required=False, allow_null=True)
    settings = RecoSettingsSerializer(required=False, default=dict)

    def validate_settings(self, value):
        s = RecoSettingsSerializer(data=value or {})
        s.is_valid(raise_exception=True)
        return s.validated_data


class TransportAuthSerializer(serializers.Serializer):
    api_key = serializers.CharField()


class TransportMenusSerializer(serializers.Serializer):
    api_key = serializers.CharField()
    organization_id = serializers.CharField()
