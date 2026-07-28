# BLS CES Download Diagnosis

Generated: 2026-07-28T16:31:59+00:00

## Previous failure
- Previous code used `download.bls.gov` with lowercase `/sm/` URLs.
- The failed URL recorded in `download_log.txt` was `https://download.bls.gov/pub/time.series/sm/sm.series`.
- The code used HTTPS GET through Python `urllib.request` with a descriptive User-Agent.
- The pipeline code did not use FTP, the BLS API, browser automation, or `www.bls.gov` for the bulk file.
- The previous response was HTTP 403 Forbidden and no valid text body was saved.

## Repaired source
- Repaired code uses the official case-sensitive bulk directory `https://download.bls.gov/pub/time.series/SM/`.
- The downloader uses GET only; no HEAD request is issued by the pipeline.
- Python streamed GET is attempted first, followed by `curl.exe` fallback with `shell=False`.

## Attempts
### Attempt 1: python_http https://download.bls.gov/pub/time.series/SM/sm.area
- method: GET
- used HEAD: False
- URL case: uppercase /SM/
- request headers: `{"Accept": "text/plain, application/octet-stream, */*", "Connection": "close", "User-Agent": "DLRHCS-replication-housing-audit/1.0 (academic reproducibility workflow)"}`
- redirect history: `[]`
- response status: 403
- response headers: `{"Alt-Svc": "h3=\":443\"; ma=93600", "Cache-control": "no-cache, no-store, must-revalidate", "Connection": "close", "Content-Length": "1325", "Content-Type": "text/html", "Date": "Tue, 28 Jul 2026 16:31:36 GMT", "Expires": "0", "Mime-Version": "1.0", "Pragma": "no-cache", "Server": "AkamaiGHost"}`
- response body prefix: `<!DOCTYPE HTML> <html lang="en-us">                      <head>  <meta http-equiv="Content-Type" content="text/html; charset=utf-8" /> <title>Access Denied</title> </head>   <style type="text/css">     .centerDiv     {       width: 60%;       height:200px;       margin: 0 auto;       background-colo`
- validation: HTTP Error 403: Forbidden
- content looked like HTML/access denied: 

### Attempt 2: python_http https://download.bls.gov/pub/time.series/SM/sm.area
- method: GET
- used HEAD: False
- URL case: uppercase /SM/
- request headers: `{"Accept": "text/plain, application/octet-stream, */*", "Connection": "close", "User-Agent": "DLRHCS-replication-housing-audit/1.0 (academic reproducibility workflow)"}`
- redirect history: `[]`
- response status: 403
- response headers: `{"Alt-Svc": "h3=\":443\"; ma=93600", "Cache-control": "no-cache, no-store, must-revalidate", "Connection": "close", "Content-Length": "1325", "Content-Type": "text/html", "Date": "Tue, 28 Jul 2026 16:31:38 GMT", "Expires": "0", "Mime-Version": "1.0", "Pragma": "no-cache", "Server": "AkamaiGHost"}`
- response body prefix: `<!DOCTYPE HTML> <html lang="en-us">                      <head>  <meta http-equiv="Content-Type" content="text/html; charset=utf-8" /> <title>Access Denied</title> </head>   <style type="text/css">     .centerDiv     {       width: 60%;       height:200px;       margin: 0 auto;       background-colo`
- validation: HTTP Error 403: Forbidden
- content looked like HTML/access denied: 

### Attempt 3: python_http https://download.bls.gov/pub/time.series/SM/sm.area
- method: GET
- used HEAD: False
- URL case: uppercase /SM/
- request headers: `{"Accept": "text/plain, application/octet-stream, */*", "Connection": "close", "User-Agent": "DLRHCS-replication-housing-audit/1.0 (academic reproducibility workflow)"}`
- redirect history: `[]`
- response status: 403
- response headers: `{"Alt-Svc": "h3=\":443\"; ma=93600", "Cache-control": "no-cache, no-store, must-revalidate", "Connection": "close", "Content-Length": "1325", "Content-Type": "text/html", "Date": "Tue, 28 Jul 2026 16:31:43 GMT", "Expires": "0", "Mime-Version": "1.0", "Pragma": "no-cache", "Server": "AkamaiGHost"}`
- response body prefix: `<!DOCTYPE HTML> <html lang="en-us">                      <head>  <meta http-equiv="Content-Type" content="text/html; charset=utf-8" /> <title>Access Denied</title> </head>   <style type="text/css">     .centerDiv     {       width: 60%;       height:200px;       margin: 0 auto;       background-colo`
- validation: HTTP Error 403: Forbidden
- content looked like HTML/access denied: 

### Attempt 4: python_http https://download.bls.gov/pub/time.series/SM/sm.area
- method: GET
- used HEAD: False
- URL case: uppercase /SM/
- request headers: `{"Accept": "text/plain, application/octet-stream, */*", "Connection": "close", "User-Agent": "DLRHCS-replication-housing-audit/1.0 (academic reproducibility workflow)"}`
- redirect history: `[]`
- response status: 403
- response headers: `{"Alt-Svc": "h3=\":443\"; ma=93600", "Cache-control": "no-cache, no-store, must-revalidate", "Connection": "close", "Content-Length": "1325", "Content-Type": "text/html", "Date": "Tue, 28 Jul 2026 16:31:51 GMT", "Expires": "0", "Mime-Version": "1.0", "Pragma": "no-cache", "Server": "AkamaiGHost"}`
- response body prefix: `<!DOCTYPE HTML> <html lang="en-us">                      <head>  <meta http-equiv="Content-Type" content="text/html; charset=utf-8" /> <title>Access Denied</title> </head>   <style type="text/css">     .centerDiv     {       width: 60%;       height:200px;       margin: 0 auto;       background-colo`
- validation: HTTP Error 403: Forbidden
- content looked like HTML/access denied: 

### Attempt 5: curl_fallback https://download.bls.gov/pub/time.series/SM/sm.area
- method: GET
- used HEAD: False
- URL case: uppercase /SM/
- request headers: `{"Accept": "text/plain, application/octet-stream, */*", "Connection": "close", "User-Agent": "DLRHCS-replication-housing-audit/1.0 (academic reproducibility workflow)"}`
- redirect history: `[]`
- response status: 22
- response headers: `{}`
- validation: curl failed
- content looked like HTML/access denied: 

