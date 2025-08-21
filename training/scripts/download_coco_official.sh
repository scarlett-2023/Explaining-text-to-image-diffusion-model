#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:-data/raw/coco}"
IMG_DIR="${OUT_ROOT}/images"
ANN_DIR="${OUT_ROOT}/annotations"

mkdir -p "${IMG_DIR}" "${ANN_DIR}"

download() {
  URL="$1"
  OUT="$2"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 16 -s 16 -c -o "$(basename "$OUT")" -d "$(dirname "$OUT")" "$URL"
  else
    wget -c -O "$OUT" "$URL"
  fi
}

echo "[1/3] Download train2017.zip (~18GB)"
download "http://images.cocodataset.org/zips/train2017.zip" "${OUT_ROOT}/train2017.zip"

echo "[2/3] Download val2017.zip (~1GB)"
download "http://images.cocodataset.org/zips/val2017.zip" "${OUT_ROOT}/val2017.zip"

echo "[3/3] Download annotations_trainval2017.zip (~250MB)"
download "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "${OUT_ROOT}/annotations_trainval2017.zip"

echo "Unzipping train2017.zip ..."
unzip -q -o "${OUT_ROOT}/train2017.zip" -d "${OUT_ROOT}"
# 结果为 ${OUT_ROOT}/train2017
mkdir -p "${IMG_DIR}"
if [ -d "${OUT_ROOT}/train2017" ]; then
  mv -f "${OUT_ROOT}/train2017" "${IMG_DIR}/" 2>/dev/null || true
fi

echo "Unzipping val2017.zip ..."
unzip -q -o "${OUT_ROOT}/val2017.zip" -d "${OUT_ROOT}"
# 结果为 ${OUT_ROOT}/val2017
if [ -d "${OUT_ROOT}/val2017" ]; then
  mv -f "${OUT_ROOT}/val2017" "${IMG_DIR}/" 2>/dev/null || true
fi

echo "Unzipping annotations ..."
unzip -q -o "${OUT_ROOT}/annotations_trainval2017.zip" -d "${OUT_ROOT}"
# 结果为 ${OUT_ROOT}/annotations/*.json
if [ -d "${OUT_ROOT}/annotations" ]; then
  # 已在目标位置
  :
elif [ -d "${OUT_ROOT}/annotations_trainval2017" ]; then
  mv -f "${OUT_ROOT}/annotations_trainval2017" "${ANN_DIR}"
else
  mkdir -p "${ANN_DIR}"
  mv -f "${OUT_ROOT}"/annotations*.json "${ANN_DIR}/" 2>/dev/null || true
fi

# 校验关键文件
REQ_FILES=(
  "${IMG_DIR}/train2017"
  "${IMG_DIR}/val2017"
  "${ANN_DIR}/captions_train2017.json"
  "${ANN_DIR}/captions_val2017.json"
)
for f in "${REQ_FILES[@]}"; do
  if [ ! -e "$f" ]; then
    echo "Missing required path: $f"
    exit 1
  fi
done

echo "Done. COCO is ready under ${OUT_ROOT}"