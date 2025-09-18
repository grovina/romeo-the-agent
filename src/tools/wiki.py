import json
import urllib.parse
import urllib.request
from typing import Any

from . import base


def wikipedia_search(query: str) -> dict[str, Any]:
    """Search Wikipedia and return a page summary."""
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"
        
        request = urllib.request.Request(url)
        request.add_header('User-Agent', 'Romeo-Agent/1.0 (https://github.com/example/romeo-agent)')
        
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read().decode('utf-8'))
            return {
                "title": data.get("title", ""),
                "summary": data.get("extract", ""),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", "")
            }
    except Exception as e:
        return {"error": f"Wikipedia search failed: {str(e)}"}


class WikiTool(base.Tool):
    name = "wiki"

    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Search Wikipedia and get article summary",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for on Wikipedia"}
                    },
                    "required": ["query"],
                },
            },
        }

    def run(self, kwargs):
        query = kwargs["query"]
        print(f"[WIKI] Searching for `{query}`...")
        result = wikipedia_search(query)
        print(f"[WIKI] Result:\n{json.dumps(result, indent=2)}")
        return result
