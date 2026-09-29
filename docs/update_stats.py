#!/usr/bin/env python3
import json
import urllib.request
from datetime import UTC, datetime


def fetch_pypi_info():
    try:
        with urllib.request.urlopen('https://pypi.org/pypi/hypernix/json', timeout=10) as resp:  # nosec B310 - a fixed https URL
            data = json.loads(resp.read().decode())
            version = data.get('info', {}).get('version', 'unknown')
    except Exception as e:
        print(f"Error fetching PyPI info: {e}")
        version = 'unknown'
    return version

def fetch_pypistats_recent():
    try:
        with urllib.request.urlopen('https://pypistats.org/api/packages/hypernix/recent', timeout=10) as resp:  # nosec B310 - a fixed https URL
            data = json.loads(resp.read().decode())
            recent = data.get('data', {})
            return {
                'last_day': recent.get('last_day', 0),
                'last_week': recent.get('last_week', 0),
                'last_month': recent.get('last_month', 0)
            }
    except Exception as e:
        print(f"Error fetching recent stats: {e}")
        return {'last_day': 0, 'last_week': 0, 'last_month': 0}

def fetch_pypistats_overall():
    try:
        with urllib.request.urlopen('https://pypistats.org/api/packages/hypernix/overall', timeout=10) as resp:  # nosec B310 - a fixed https URL
            data = json.loads(resp.read().decode())
            total = data.get('data', {}).get('total_downloads', 0)
            return total
    except Exception as e:
        print(f"Error fetching overall stats: {e}")
        return 0

def main():
    version = fetch_pypi_info()
    recent = fetch_pypistats_recent()
    total = fetch_pypistats_overall()
    
    result = {
        'version': version,
        'total_downloads': total,
        'last_day': recent['last_day'],
        'last_week': recent['last_week'],
        'last_month': recent['last_month'],
        'updated_at': datetime.now(UTC).isoformat().replace('+00:00', 'Z')
    }
    
    with open('/workspace/docs/v1.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
