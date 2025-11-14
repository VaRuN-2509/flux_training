import tomllib

with open("pyproject.toml", "rb") as f:
    data = tomllib.load(f)

deps = data.get("project", {}).get("dependencies", [])
extras = data.get("project", {}).get("optional-dependencies", {}).get("all", [])
all_deps = deps + extras

print(all_deps)
