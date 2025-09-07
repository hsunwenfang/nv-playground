


curl -s http://localhost:80
00/chat -X POST -H 'Content-Type: application/json' -d '{\"message\":\"hello\"}'
{"detail":[{"type":"json_invalid","loc":["body",1],"msg":"JSON decode error","input":{},"ctx":{"error":"Expecting property name enclosed in double quotes"}}]}

# Seccussful request for tiny-chat service

## get model name
```http
GET /healthz HTTP/1.1
Host: localhost:8000
Content-Type: application/json
```

## chat request
```http
POST /chat HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{"message":"Is Sushiro the top sushi restaurant in Japan?"}
```