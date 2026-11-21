import re
import time
import fsspec
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

# --- Configuration ---
HF_REPO = "ayush5218/AT"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"

# Initialize FastAPI
app = FastAPI(title="Smart Universal Search Engine")

# Initialize fsspec HTTP filesystem (handles HF redirects flawlessly)
fs = fsspec.filesystem("http")

# --- PHONE RANGE ROUTER (Author's Method) ---
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

def clean_results(table):
    if table is None or table.num_rows == 0:
        return []
    records = table.to_pylist()
    cleaned = []
    for rec in records:
        clean_rec = {k: v for k, v in rec.items() if v is not None and str(v).strip().lower() not in ['null', 'none', '']}
        cleaned.append(clean_rec)
    return cleaned

# --- API Endpoint ---
@app.get("/search")
def search(q: str = Query(..., description="Search query (Phone, Email, or Name)")):
    start_time = time.time()
    q = q.strip()
    if not q:
        return JSONResponse(status_code=400, content={"error": "Empty query"})

    results = {}
    clean_phone = re.sub(r'\D', '', q)

    try:
        # 1. Phone Search (Blazing Fast Predicate Pushdown)
        if len(clean_phone) == 10 and clean_phone == q.replace(" ", "").replace("-", ""):
            target_filename = get_phone_target_file(clean_phone)
            if not target_filename:
                return {"error": "Number out of range"}
                
            url = f"{HF_BASE_URL}/A_index/{target_filename}"
            
            # PyArrow will ONLY download the row groups that match this phone number
            table = pq.read_table(
                url, 
                columns=['mobile', 'name', 'email', 'address', 'carrier'], 
                filters=[('mobile', '=', int(clean_phone))],
                filesystem=fs
            )
            if table.num_rows > 0:
                results["Airtel_Master"] = clean_results(table)
                
        # 2. Email Search (Case-Insensitive)
        elif '@' in q:
            email = q.lower()
            bucket = get_alphabet_bucket_number(email)
            
            if bucket == "overflow":
                url = f"{HF_BASE_URL}/E_index/E_overflow.parquet"
                result_key = "E_Overflow"
            else:
                url = f"{HF_BASE_URL}/E_index/E_{bucket}.parquet"
                result_key = f"E_{bucket}"

            # Read only needed columns to save bandwidth
            table = pq.read_table(url, columns=['email', 'name', 'mobile'], filesystem=fs)
            
            # pc.match_substring ignores case natively in C++
            col = table['email'].cast(pa.string())
            mask = pc.match_substring(col, email, ignore_case=True)
            mask = pc.fill_null(mask, False)
            
            filtered = table.filter(mask)
            if filtered.num_rows > 0:
                results[result_key] = clean_results(filtered)
                
        # 3. Name Search (Case-Insensitive Substring)
        else:
            name = q.lower()
            bucket = get_alphabet_bucket_number(name)
            
            if bucket == "overflow":
                url = f"{HF_BASE_URL}/D_index/D_overflow.parquet"
                result_key = "D_Overflow"
            else:
                url = f"{HF_BASE_URL}/D_index/D_{bucket}.parquet"
                result_key = f"D_{bucket}"

            table = pq.read_table(url, columns=['name', 'mobile', 'email'], filesystem=fs)
            col = table['name'].cast(pa.string())
            mask = pc.match_substring(col, name, ignore_case=True)
            mask = pc.fill_null(mask, False)
            
            filtered = table.filter(mask)
            if filtered.num_rows > 0:
                results[result_key] = clean_results(filtered)

    except Exception as e:
        # Catch file-not-found or HTTP errors silently
        pass

    exec_time = (time.time() - start_time) * 1000
    
    return {
        "query": q,
        "results": results if results else None,
        "execution_time_ms": round(exec_time, 2)
    }
