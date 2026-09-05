#!/usr/bin/env bash
# Exercise the live API end-to-end: legit merchant, then a 3-member fraud ring.
set -euo pipefail
API="http://127.0.0.1:8000/api/v1"

submit() {
  curl -s -X POST "$API/merchants/assess" \
    -F "business_name=$1"        -F "business_type=$2" \
    -F "owner_name=$3"           -F "owner_pan=$4" \
    -F "owner_email=$5"          -F "owner_phone=$6" \
    -F "bank_account=$7"         -F "bank_ifsc=$8" \
    -F "registered_address=$9"   -F "device_fingerprint=${10}" \
    -F "ip_address=${11}"
  echo
}

echo "=== 1. Legitimate merchant ==="
submit "Priya's Boutique" retail "Priya Sharma" BKAPS1234F \
  priya@priyaboutique.in +919845012345 50100123456789 HDFC0001234 \
  "Shop 12, MG Road, Bangalore 560001" fp_legit_a1 103.21.58.9

echo "=== 2. Ring member 1 (looks fine alone) ==="
submit "Best Electronics Hub" retail "Amit Verma" DEFAV9876H \
  best@tempmail.com +917777666655 98765432100 ICIC0001234 \
  "123 Industrial Area Phase 2, Noida 201301" fp_ring_01 45.67.89.101

echo "=== 3. Ring member 2 (clustering fires) ==="
submit "Super Mobile World" retail "Vikram Singh" GHIVS4321J \
  super@guerrillamail.com +917777666656 98765432100 ICIC0001234 \
  "124 Industrial Area Phase 2, Noida 201301" fp_ring_01 45.67.89.101

echo "=== 4. Ring member 3 (linked to a rejected merchant) ==="
submit "Prime Gadget Store" retail "Rohit Yadav" JKLRY8765K \
  prime@mailinator.com +917777666657 98765432100 ICIC0001234 \
  "125 Industrial Area Phase 2, Noida 201301" fp_ring_01 45.67.89.101

echo "=== 5. Synthetic identity ==="
submit "Test Store Enterprise" services "John Test" QWERT1111Z \
  john.test123456@yahoo.com +916111111111 11111111111 UTIB0001111 \
  "Test Address" fp_synth_z9 8.8.8.8
