// ============================================================
// SCHEMA Neo4j 5.x cho KG Luật 41/2024/QH15
// Chạy: cypher-shell -u neo4j -p <pw> -f schema/schema.cypher
// Idempotent — chạy lại an toàn.
// ============================================================

// ---------- 1. UNIQUENESS CONSTRAINTS cho mọi label ----------
// Đảm bảo mỗi entity có ID duy nhất → dedup khi MERGE và truy ngược không nhập nhằng.

CREATE CONSTRAINT law_id            IF NOT EXISTS FOR (n:Law)            REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT chapter_id        IF NOT EXISTS FOR (n:Chapter)        REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT section_id        IF NOT EXISTS FOR (n:Section)        REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT article_id        IF NOT EXISTS FOR (n:Article)        REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT clause_id         IF NOT EXISTS FOR (n:Clause)         REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT point_id          IF NOT EXISTS FOR (n:Point)          REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT table_id          IF NOT EXISTS FOR (n:Table)          REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT concept_id        IF NOT EXISTS FOR (n:LegalConcept)   REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT subject_id        IF NOT EXISTS FOR (n:Subject)        REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT organization_id   IF NOT EXISTS FOR (n:Organization)   REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT role_id           IF NOT EXISTS FOR (n:Role)           REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT benefit_id        IF NOT EXISTS FOR (n:Benefit)        REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT condition_id      IF NOT EXISTS FOR (n:Condition)      REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT obligation_id     IF NOT EXISTS FOR (n:Obligation)     REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT right_id          IF NOT EXISTS FOR (n:Right)          REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT prohib_id         IF NOT EXISTS FOR (n:ProhibitedAct)  REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT fund_id           IF NOT EXISTS FOR (n:Fund)           REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT external_law_id   IF NOT EXISTS FOR (n:ExternalLaw)    REQUIRE n.id IS UNIQUE;

// ---------- 2. EXISTENCE CONSTRAINTS — bảo đảm PROVENANCE ----------
// Mọi structural node phải có `text` (Article/Clause/Point) → tự nó là nguồn.

CREATE CONSTRAINT article_text_exists IF NOT EXISTS FOR (n:Article) REQUIRE n.text IS NOT NULL;
CREATE CONSTRAINT clause_text_exists  IF NOT EXISTS FOR (n:Clause)  REQUIRE n.text IS NOT NULL;
CREATE CONSTRAINT point_text_exists   IF NOT EXISTS FOR (n:Point)   REQUIRE n.text IS NOT NULL;

// Mọi semantic node phải có mentioned_in (lưu dạng list JSON string vì Neo4j list constraints hạn chế)
// → Sẽ validate ở tầng B4 Python; ở Cypher chỉ ràng buộc property tồn tại.
CREATE CONSTRAINT subject_prov     IF NOT EXISTS FOR (n:Subject)       REQUIRE n.mentioned_in IS NOT NULL;
CREATE CONSTRAINT benefit_prov     IF NOT EXISTS FOR (n:Benefit)       REQUIRE n.mentioned_in IS NOT NULL;
CREATE CONSTRAINT organization_prov IF NOT EXISTS FOR (n:Organization) REQUIRE n.mentioned_in IS NOT NULL;
CREATE CONSTRAINT role_prov        IF NOT EXISTS FOR (n:Role)          REQUIRE n.mentioned_in IS NOT NULL;
CREATE CONSTRAINT concept_prov     IF NOT EXISTS FOR (n:LegalConcept)  REQUIRE n.defined_in IS NOT NULL;
CREATE CONSTRAINT condition_prov   IF NOT EXISTS FOR (n:Condition)     REQUIRE n.mentioned_in IS NOT NULL;
CREATE CONSTRAINT obligation_prov  IF NOT EXISTS FOR (n:Obligation)    REQUIRE n.mentioned_in IS NOT NULL;
CREATE CONSTRAINT right_prov       IF NOT EXISTS FOR (n:Right)         REQUIRE n.mentioned_in IS NOT NULL;
CREATE CONSTRAINT prohib_prov      IF NOT EXISTS FOR (n:ProhibitedAct) REQUIRE n.mentioned_in IS NOT NULL;
CREATE CONSTRAINT fund_prov        IF NOT EXISTS FOR (n:Fund)          REQUIRE n.mentioned_in IS NOT NULL;

// Mọi cạnh semantic + reference phải có source_clause (Clause.id gốc) — đảm bảo trace ngược.
// Neo4j 5 cho phép relationship property existence constraint:
CREATE CONSTRAINT ref_source       IF NOT EXISTS FOR ()-[r:REFERENCES]-()       REQUIRE r.source_clause IS NOT NULL;
CREATE CONSTRAINT cite_ext_source  IF NOT EXISTS FOR ()-[r:CITES_EXTERNAL]-()   REQUIRE r.source_clause IS NOT NULL;
CREATE CONSTRAINT amends_source    IF NOT EXISTS FOR ()-[r:AMENDS]-()           REQUIRE r.source_clause IS NOT NULL;
CREATE CONSTRAINT repeals_source   IF NOT EXISTS FOR ()-[r:REPEALS]-()          REQUIRE r.source_clause IS NOT NULL;
CREATE CONSTRAINT replaces_source  IF NOT EXISTS FOR ()-[r:REPLACES]-()         REQUIRE r.source_clause IS NOT NULL;
CREATE CONSTRAINT trans_source     IF NOT EXISTS FOR ()-[r:TRANSITIONS_FROM]-() REQUIRE r.source_clause IS NOT NULL;

CREATE CONSTRAINT entitled_src     IF NOT EXISTS FOR ()-[r:ENTITLED_TO]-()      REQUIRE r.source_clause IS NOT NULL;
CREATE CONSTRAINT entitled_txt     IF NOT EXISTS FOR ()-[r:ENTITLED_TO]-()      REQUIRE r.source_text   IS NOT NULL;
CREATE CONSTRAINT oblig_src        IF NOT EXISTS FOR ()-[r:HAS_OBLIGATION]-()   REQUIRE r.source_clause IS NOT NULL;
CREATE CONSTRAINT oblig_txt        IF NOT EXISTS FOR ()-[r:HAS_OBLIGATION]-()   REQUIRE r.source_text   IS NOT NULL;
CREATE CONSTRAINT right_src        IF NOT EXISTS FOR ()-[r:HAS_RIGHT]-()        REQUIRE r.source_clause IS NOT NULL;
CREATE CONSTRAINT right_txt        IF NOT EXISTS FOR ()-[r:HAS_RIGHT]-()        REQUIRE r.source_text   IS NOT NULL;
CREATE CONSTRAINT applies_src      IF NOT EXISTS FOR ()-[r:APPLIES_TO]-()       REQUIRE r.source_clause IS NOT NULL;
CREATE CONSTRAINT requires_src     IF NOT EXISTS FOR ()-[r:REQUIRES]-()         REQUIRE r.source_clause IS NOT NULL;
CREATE CONSTRAINT resp_src         IF NOT EXISTS FOR ()-[r:RESPONSIBLE_FOR]-()  REQUIRE r.source_clause IS NOT NULL;
CREATE CONSTRAINT prohib_by_src    IF NOT EXISTS FOR ()-[r:PROHIBITED_BY]-()    REQUIRE r.source_clause IS NOT NULL;

// ---------- 3. RANGE INDEXES cho query hay dùng ----------
CREATE INDEX article_number   IF NOT EXISTS FOR (n:Article)   ON (n.number);
CREATE INDEX chapter_number   IF NOT EXISTS FOR (n:Chapter)   ON (n.number);
CREATE INDEX subject_name     IF NOT EXISTS FOR (n:Subject)   ON (n.name);
CREATE INDEX benefit_name     IF NOT EXISTS FOR (n:Benefit)   ON (n.name);
CREATE INDEX benefit_category IF NOT EXISTS FOR (n:Benefit)   ON (n.category);

// ---------- 4. FULL-TEXT INDEXES (cho keyword fallback trong RAG) ----------
CREATE FULLTEXT INDEX clause_fulltext IF NOT EXISTS
  FOR (n:Clause|Article|Point) ON EACH [n.text, n.title];

// ---------- 5. VECTOR INDEXES (BGE-M3, 1024-d, cosine) ----------
// Cho 3 cấp độ truy hồi semantic search.
CREATE VECTOR INDEX article_vec IF NOT EXISTS
  FOR (n:Article) ON n.embedding
  OPTIONS {indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }};

CREATE VECTOR INDEX clause_vec IF NOT EXISTS
  FOR (n:Clause) ON n.embedding
  OPTIONS {indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }};

CREATE VECTOR INDEX point_vec IF NOT EXISTS
  FOR (n:Point) ON n.embedding
  OPTIONS {indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }};
