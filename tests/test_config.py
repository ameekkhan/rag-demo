from app.config import Settings


def test_default_models_are_configured():
    settings = Settings()
    assert settings.llm_model
    assert settings.embedding_model
