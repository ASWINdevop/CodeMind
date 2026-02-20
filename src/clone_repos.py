import os
import subprocess
import requests
from src.config import REPOS_DIR

def get_available_repos(username: str) -> list[str]:
    """Fetches a list of repository names for a given GitHub user."""
    # Confidence Level: 95%. The API will return data unless rate-limited or the user doesn't exist.
    url = f"https://api.github.com/users/{username}/repos?per_page=100"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Strictly fail on non-200 HTTP responses
    except requests.exceptions.RequestException as e:
        print(f"API Request Failed: {e}")
        return []

    repos_data = response.json()
    return [repo['name'] for repo in repos_data]

def clone_selected_repos(username: str, selected_repos: list[str]):
    """Clones or pulls specific repositories selected by the user."""
    if not selected_repos:
        print("No repositories selected for ingestion.")
        return

    os.makedirs(REPOS_DIR, exist_ok=True)
    
    # We must construct the clone URL manually to avoid hitting the API a second time
    for repo_name in selected_repos:
        clone_url = f"https://github.com/{username}/{repo_name}.git"
        target_path = os.path.join(REPOS_DIR, repo_name)

        if os.path.exists(target_path):
            print(f"Directory exists. Executing git pull for {repo_name}...")
            # check=False prevents one failed pull (e.g., uncommitted local changes) from crashing the loop
            subprocess.run(["git", "-C", target_path, "pull"], check=False)
        else:
            print(f"Executing git clone for {repo_name}...")
            subprocess.run(["git", "clone", clone_url, target_path], check=False)

if __name__ == "__main__":
    # CLI Testing Block
    user = "ASWINdevop"
    repos = get_available_repos(user)
    print(f"Found: {repos}")
    # Example: clone_selected_repos(user, repos[:1])