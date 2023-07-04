from locust import FastHttpUser, between, task

test_request = {
    "DIA": 13,
    "MES": 9,
    "DIANOM": "Miercoles",
    "TIPOVUELO": "N",
    "OPERA": "Grupo LATAM",
    "SIGLADES": "Arica",
    "TEMPORADAALTA": 1,
    "PERIODODIA": "noche",
}


class PerformanceTests(FastHttpUser):
    wait_time = between(1, 3)

    @task(1)
    def predict(self):
        self.client.post(
            "/predict",
            json=test_request,
        )
