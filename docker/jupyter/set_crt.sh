key_file='/ssl.key'
csr_file='/ssl.csr'
crt_file='/ssl.crt'
san_file='/san.txt'
hostname=`curl inet-ip.info`

echo "subjectAltName = DNS:${hostname}, DNS:*.localhost, DNS:localhost" > ${san_file}
openssl genrsa -out ${key_file} 2048
openssl req -out ${csr_file} -key ${key_file} -new -nodes -subj "/C=JP/ST=Tokyo/L=Ropongi/CN=${hostname}"
openssl x509 -req -days 30 -signkey ${key_file} -in ${csr_file} -out ${crt_file} -extfile ${san_file}