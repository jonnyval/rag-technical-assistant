"""Compatibility wrapper: switch Hermes to the project Yandex profile."""

from configure_provider import apply_profile


if __name__ == "__main__":
    name, profile = apply_profile("yandex_deepseek")
    print(f"Hermes profile applied: {name}; model={profile['model']}; base_url={profile['base_url']}")
