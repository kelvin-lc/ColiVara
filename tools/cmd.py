"""



uvicorn config.asgi:application --reload --host 0.0.0.0 --port 8010

python main.py --rp_api_host 0.0.0.0 --rp_serve_api

a057CEQeolH5iebfqW8rrdU3UNnTYmAV



curl --request POST \
  --url 10.112.1.133:8000/runsync \
  --header 'Content-Type: application/json' \
  --data '{"input": {"task": "query","input_data": ["hello"]}}
"""
