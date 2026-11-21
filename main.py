import re
import duckdb
import os
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

# --- Configuration ---
HF_REPO = "ayush5218/AT"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"

# --- Vercel / Lambda DuckDB Setup ---
# Vercel has a read-only filesystem. DuckDB needs a writable dir to unpack httpfs.
os.makedirs("/tmp/duckdb_ext", exist_ok=True)
con = duckdb.connect()
con.execute("SET extension_directory='/tmp/duckdb_ext';")
con.execute("INSTALL httpfs;")
con.execute("LOAD httpfs;")

# Initialize FastAPI app
app = FastAPI(title="Smart Universal Search Engine")

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

def clean_dict(row):
    return {k: v for k, v in row.items() if v is not None and str(v).strip().lower() not in ['null', 'none', '']}

# --- API Endpoint ---
@app.get("/search")
def search(q: str = Query(..., description="Search query (Phone, Email, or Name)")):
    start_time = __import__('time').time()
    results = {}
    q = q.strip()
    
    if not q:
        return JSONResponse(status_code=400, content={"error": "Empty query"})

    clean_phone = re.sub(r'\D', '', q)
    
    try:
        # 1. Phone Search
        if len(clean_phone) == 10 and clean_phone == q.replace(" ", "").replace("-", ""):
            target_filename = get_phone_target_file(clean_phone)
            if not target_filename:
                return {"error": "Number out of range"}
                
            url = f"{HF_BASE_URL}/A_index/{target_filename}"
            query = "SELECT mobile, name, email, address, carrier FROM read_parquet(?) WHERE mobile = ?"
            res = con.execute(query, [url, int(clean_phone)]).fetchall()
            
            if res:
                columns = ['mobile', 'name', 'email', 'address', 'carrier']
                results["Airtel_Master"] = [clean_dict(dict(zip(columns, row))) for row in res]
                
        # 2. Email Search
        elif '@' in q:
            email = q.lower()
            bucket = get_alphabet_bucket_number(email)
            
            if bucket == "overflow":
                url = f"{HF_BASE_URL}/E_index/E_overflow.parquet"
                result_key = "E_Overflow"
            else:
                url = f"{HF_BASE_URL}/E_index/E_{bucket}.parquet"
                result_key = f"E_{bucket}"

            # ILIKE makes it case-insensitive natively in DuckDB C++
            query = "SELECT email, name, mobile FROM read_parquet(?) WHERE email ILIKE ?"
            res = con.execute(query, [url, email]).fetchall()
            
            if res:
                columns = ['email', 'name', 'mobile']
                results[result_key] = [clean_dict(dict(zip(columns, row))) for row in res]
                
        # 3. Name Search
        else:
            name = q.lower()
            bucket = get_alphabet_bucket_number(name)
            
            if bucket == "overflow":
                url = f"{HF_BASE_URL}/D_index/D_overflow.parquet"
                result_key = "D_Overflow"
            else:
                url = f"{HF_BASE_URL}/D_index/D_{bucket}.parquet"
                result_key = f"D_{bucket}"

            # % wildcards for partial name matching
            query = "SELECT name, mobile, email FROM read_parquet(?) WHERE name ILIKE ?"
            res = con.execute(query, [url, f"%{name}%"]).fetchall()
            
            if res:
                columns = ['name', 'mobile', 'email']
                results[result_key] = [clean_dict(dict(zip(columns, row))) for row in res]

    except Exception as e:
        # Catch network issues or file-not-found issues silently, or log them
        pass

    exec_time = (__import__('time').time() - start_time) * 1000
    
    return {
        "query": q,
        "results": results if results else None,
        "execution_time_ms": round(exec_time, 2)
    }
