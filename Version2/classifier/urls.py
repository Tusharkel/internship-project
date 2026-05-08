from django.urls import path
from classifier.views import (
    ClassifyTextView,
    EmailListView,
    EmailDetailView,
    FlagEmailView,
    StatsView,
)

urlpatterns = [
    path("classify/",                   ClassifyTextView.as_view(),  name="classify"),
    path("emails/",                     EmailListView.as_view(),     name="email-list"),
    path("emails/<str:email_id>/",      EmailDetailView.as_view(),   name="email-detail"),
    path("emails/<str:email_id>/flag/", FlagEmailView.as_view(),     name="email-flag"),
    path("stats/",                      StatsView.as_view(),         name="stats"),
]