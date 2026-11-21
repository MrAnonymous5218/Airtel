from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow as pa
from huggingface_hub import HfFileSystem
import time
import json
import re

app = FastAPI(title="Airtel Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
HF_REPO = "Anonymous-5218/AT" 
global_fs = HfFileSystem(token=False) 
dataset_cache = {}

# --- PHONE RANGE ROUTER ---
PHONE_LIMITS = [
    (6230309204, 7051894468, "A_1.parquet"), (7051894476, 7250427787, "A_2.parquet"),
    (7250427792, 7348939102, "A_3.parquet"), (7348939103, 7461906080, "A_4.parquet"),
    (7461906110, 7600887240, "A_5.parquet"), (7600887242, 7718948995, "A_6.parquet"),
    (7718948998, 7893273284, "A_7.parquet"), (7893273289, 8011313654, "A_8.parquet"),
    (8011313655, 8128990920, "A_9.parquet"), (8128990929, 8264574462, "A_10.parquet"),
    (8264574465, 8427455703, "A_11.parquet"), (8427455710, 8530364239, "A_12.parquet"),
    (8530364242, 8800112391, "A_13.parquet"), (8800112393, 8897217348, "A_14.parquet"),
    (8897217355, 9003800304, "A_15.parquet"), (9003800305, 9121159099, "A_16.parquet"),
    (9121159105, 9179948267, "A_17.parquet"), (9179948277, 9546485222, "A_18.parquet"),
    (9546485229, 9571032089, "A_19.parquet"), (9571032091, 9601128047, "A_20.parquet"),
    (9601128049, 9633280256, "A_21.parquet"), (9633280257, 9665639812, "A_22.parquet"),
    (9665639817, 9701973901, "A_23.parquet"), (9701973903, 9746833234, "A_24.parquet"),
    (9746833241, 9784425260, "A_25.parquet"), (9784425263, 9805510003, "A_26.parquet"),
    (9805510005, 9821223534, "A_27.parquet"), (9821223535, 9860550287, "A_28.parquet"),
    (9860550288, 9872387461, "A_29.parquet"), (9872387467, 9893643456, "A_30.parquet"),
    (9893643457, 9906505266, "A_31.parquet"), (9906505267, 9931724504, "A_32.parquet"),
    (9931724505, 9952250036, "A_33.parquet"), (9952250042, 9963182899, "A_34.parquet"),
    (9963182908, 9975344489, "A_35.parquet"), (9975344492, 9995185237, "A_36.parquet"),
    (9995185241, 9998999999, "A_37.parquet")
]

def get_phone_target_file(phone_number):
    phone_int = int(phone_number)
    for min_num, max_num, filename in PHONE_LIMITS:
        if min_num <= phone_int <= max_num:
            return filename
    return None

def get_alphabet_bucket_number(text):
    if not text: return "overflow"
    first_char = text[0].lower()
    if 'a' <= first_char <= 'z':
        return str(ord(first_char) - ord('a') + 1)
    return "overflow"

def get_dataset(file_path):
    if file_path not in dataset_cache:
        dataset_cache[file_path] = ds.dataset(file_path, filesystem=global_fs, format="parquet")
    return dataset_cache[file_path]

def clean_records(table, source_name):
    if table is None or table.num_rows == 0:
        return []
    records = table.to_pylist()
    cleaned = []
    for rec in records:
        clean_rec = {k: v for k, v in rec.items() if v is not None and str(v).strip().lower() not in ['null', 'none', '']}
        clean_rec["source"] = source_name
        cleaned.append(clean_rec)
    return cleaned

@app.get("/")
def root():
    return {"status": "online", "message": "Airtel Search API is Live."}

@app.get("/search")
def search(q: str = Query(..., description="Search query (Phone, Email, or Name)")):
    start_time = time.time()
    q = q.strip()
    if not q:
        return Response(content=json.dumps({"error": "Empty query"}, indent=4), media_type="application/json", status_code=400)

    results = {}
    clean_phone = re.sub(r'\D', '', q)

    try:
        # 1. Phone Search
        if len(clean_phone) == 10 and clean_phone == q.replace(" ", "").replace("-", ""):
            target_filename = get_phone_target_file(clean_phone)
            if not target_filename:
                return Response(content=json.dumps({"error": "Number out of range"}, indent=4), media_type="application/json", status_code=400)
                
            file_path = f"datasets/{HF_REPO}/A_index/{target_filename}"
            dataset = get_dataset(file_path)
            
            filter_expr = ds.field('mobile') == int(clean_phone)
            table = dataset.to_table(filter=filter_expr)
            
            if table.num_rows > 0:
                results["Airtel_Master"] = clean_records(table, "Airtel")
                
        # 2. Email Search
        elif '@' in q:
            email = q.lower()
            bucket = get_alphabet_bucket_number(email)
            
            target_filename = f"E_overflow.parquet" if bucket == "overflow" else f"E_{bucket}.parquet"
            result_key = "E_Overflow" if bucket == "overflow" else f"E_{bucket}"
            
            file_path = f"datasets/{HF_REPO}/E_index/{target_filename}"
            dataset = get_dataset(file_path)
            
            table = dataset.to_table(columns=['email', 'name', 'mobile'])
            col = table['email'].cast(pa.string())
            mask = pc.match_substring(col, email, ignore_case=True)
            mask = pc.fill_null(mask, False)
            
            filtered = table.filter(mask)
            if filtered.num_rows > 0:
                results[result_key] = clean_records(filtered, "Airtel")
                
        # 3. Name Search
        else:
            name = q.lower()
            bucket = get_alphabet_bucket_number(name)
            
            target_filename = f"D_overflow.parquet" if bucket == "overflow" else f"D_{bucket}.parquet"
            result_key = "D_Overflow" if bucket == "overflow" else f"D_{bucket}"
            
            file_path = f"datasets/{HF_REPO}/D_index/{target_filename}"
            dataset = get_dataset(file_path)
            
            table = dataset.to_table(columns=['name', 'mobile', 'email'])
            col = table['name'].cast(pa.string())
            mask = pc.match_substring(col, name, ignore_case=True)
            mask = pc.fill_null(mask, False)
            
            filtered = table.filter(mask)
            if filtered.num_rows > 0:
                results[result_key] = clean_records(filtered, "Airtel")

    except Exception:
        pass

    exec_time = (time.time() - start_time) * 1000
    
    # Format the output exactly as requested
    response_data = {
        "credits": "Mr Anonymous",
        "query": q,
        "source": "Airtel",
        "execution_time_ms": round(exec_time, 2),
        "results": results if results else None
    }
    
    # Force indent=4 permanently. No URL parameter needed.
    return Response(content=json.dumps(response_data, indent=4, default=str), media_type="application/json")
