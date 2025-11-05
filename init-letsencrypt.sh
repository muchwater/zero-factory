#!/bin/bash

# Let's Encrypt SSL 인증서 초기화 스크립트
# 참조: https://github.com/wmnnd/nginx-certbot

set -e

domains=(zeromap.store www.zeromap.store)
rsa_key_size=4096
data_path="./certbot"
email="admin@zeromap.store" # 실제 이메일로 변경하세요
staging=0 # Set to 1 for testing to avoid rate limits

if [ -d "$data_path" ]; then
  read -p "Existing data found for $domains. Continue and replace existing certificate? (y/N) " decision
  if [ "$decision" != "Y" ] && [ "$decision" != "y" ]; then
    exit
  fi
fi

# Nginx가 실행 중이면 중지
echo "### Stopping existing Nginx container ..."
docker-compose stop nginx 2>/dev/null || true

# 더미 인증서 생성 (Nginx가 시작될 수 있도록)
echo "### Creating dummy certificate for $domains ..."
path="/etc/letsencrypt/live/${domains[0]}"
mkdir -p "$data_path/conf/live/${domains[0]}"
docker-compose run --rm --entrypoint "\
  openssl req -x509 -nodes -newkey rsa:$rsa_key_size -days 1\
    -keyout '$path/privkey.pem' \
    -out '$path/fullchain.pem' \
    -subj '/CN=localhost'" certbot
echo

# Nginx 시작
echo "### Starting nginx ..."
docker-compose up --force-recreate -d nginx
echo

# 더미 인증서 삭제
echo "### Deleting dummy certificate for $domains ..."
docker-compose run --rm --entrypoint "\
  rm -rf /etc/letsencrypt/live/${domains[0]} && \
  rm -rf /etc/letsencrypt/archive/${domains[0]} && \
  rm -rf /etc/letsencrypt/renewal/${domains[0]}.conf" certbot
echo

# 실제 인증서 요청
echo "### Requesting Let's Encrypt certificate for $domains ..."
domain_args=""
for domain in "${domains[@]}"; do
  domain_args="$domain_args -d $domain"
done

# 이메일 인자 설정
case "$email" in
  "") email_arg="--register-unsafely-without-email" ;;
  *) email_arg="--email $email" ;;
esac

# staging 여부 설정
if [ $staging != "0" ]; then staging_arg="--staging"; fi

docker-compose run --rm --entrypoint "\
  certbot certonly --webroot -w /var/www/certbot \
    $staging_arg \
    $email_arg \
    $domain_args \
    --rsa-key-size $rsa_key_size \
    --agree-tos \
    --force-renewal" certbot
echo

# Nginx 재시작
echo "### Reloading nginx ..."
docker-compose exec nginx nginx -s reload

echo "### Certificate successfully obtained!"
echo "### HTTPS is now enabled for $domains"
