from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
			path("AdminLogin.html", views.AdminLogin, name="AdminLogin"),
			path("AdminLoginAction", views.AdminLoginAction, name="AdminLoginAction"),
			path("UploadDataset", views.UploadDataset, name="UploadDataset"),
			path("RunGan", views.RunGan, name="RunGan"),
			path("RunPso", views.RunPso, name="RunPso"),
			path("TrainML", views.TrainML, name="TrainML"),
			path("Predict", views.Predict, name="Predict"),
			path("PredictAction", views.PredictAction, name="PredictAction"),			
]