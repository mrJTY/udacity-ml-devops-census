from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_api_locally_get_root():
    response = client.get("/")
    assert response.status_code == 200


def test_post():
    response = client.post("/items", json={
        "name": "item",
        "tags": "tag1",
        "item_id": 0
    })
    assert response.status_code == 200


def test_post_list():
    response = client.post("/items", json={
        "name": "item",
        "tags": ["tag1", "tag2"],
        "item_id": 0
    })
    assert response.status_code == 200
