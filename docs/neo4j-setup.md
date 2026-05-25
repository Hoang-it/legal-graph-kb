# Cài đặt Neo4j Desktop (Windows)

> Mục tiêu: có một DBMS Neo4j 5.x chạy local trên `bolt://localhost:7687` để các bước B6/B7 kết nối.

## 1. Cài Neo4j Desktop

1. Tải bản Windows tại: <https://neo4j.com/download/> (yêu cầu đăng ký email, lấy activation key).
2. Cài đặt mặc định. Mở Neo4j Desktop.

## 2. Tạo DBMS local 5.x

1. Trong tab **Projects** → tạo project `legal-graph-kb`.
2. Click **Add → Local DBMS**.
   - Name: `law-41-2024`
   - Version: chọn bản **5.x mới nhất** (≥ 5.13, cần hỗ trợ vector index).
   - Password: đặt rồi LƯU LẠI (ví dụ `legalkg2025`). Sẽ điền vào `.env`.
3. Click **Start** trên DBMS vừa tạo. Đợi trạng thái chuyển sang **Active**.

## 3. Cài plugin APOC (bắt buộc) + GDS (tuỳ chọn)

1. Click DBMS `law-41-2024` → tab **Plugins**.
2. Trong mục **APOC** → click **Install**.
3. (Tuỳ chọn) **Graph Data Science Library** nếu sau này muốn chạy thuật toán đồ thị.
4. Restart DBMS sau khi cài plugin.

## 4. Lấy connection string

1. Click DBMS đang chạy → tab **Details**.
2. Mục **Bolt port**: thường là `7687`. URI: `bolt://localhost:7687`.
3. User mặc định: `neo4j`. Password: đã đặt ở bước 2.2.

## 5. Cập nhật `.env`

```dotenv
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=legalkg2025   # password bạn đã đặt
NEO4J_DATABASE=neo4j
```

## 6. Áp schema

Mở **Neo4j Browser** (click **Open** → **Neo4j Browser** trên DBMS đang chạy), copy toàn bộ nội dung `schema/schema.cypher` và paste vào ô lệnh `:source` rồi chạy. Hoặc dùng `cypher-shell`:

```powershell
# Tìm cypher-shell trong Neo4j Desktop:
# C:\Users\<bạn>\.Neo4jDesktop\relate-data\dbmss\<hash>\bin\cypher-shell.exe
& "<đường dẫn cypher-shell.exe>" -u neo4j -p legalkg2025 -f schema/schema.cypher
```

Kiểm tra constraints + indexes đã tạo:

```cypher
SHOW CONSTRAINTS;
SHOW INDEXES;
```

Phải thấy ≥ 17 UNIQUE constraints, ≥ 13 existence constraints, 3 vector index (`article_vec`, `clause_vec`, `point_vec`) và 1 fulltext (`clause_fulltext`).

## 7. Reset DB (nếu cần làm lại từ đầu)

Trong Neo4j Browser:
```cypher
MATCH (n) DETACH DELETE n;
```
Constraints/indexes vẫn giữ — không cần áp lại `schema.cypher`.
