# core/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.chat, name="chat"),
    path("ask/", views.ask, name="ask"),
    path("new/", views.new_chat, name="new_chat"),
    path("switch/<str:chat_id>/", views.switch_chat, name="switch_chat"),
    path("search_chats/", views.search_chats, name="search_chats"),
    path("feedback/", views.feedback, name="feedback"),
    path("clear_history/", views.clear_history, name="clear_history"),

    #  Sidebar item-ийн 3 цэгийн цэс
    path("chats/<str:cid>/", views.chat_item, name="chat_item"),
    path("chats/<str:cid>/rename/", views.rename_chat, name="rename_chat"),
    path("chats/<str:cid>/delete/", views.delete_chat, name="delete_chat"),
]
