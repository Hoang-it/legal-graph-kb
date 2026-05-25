# Báo cáo trích xuất KG — Luật 41/2024/QH15

## Tổng quan

- **Tổng số node:** 1,334
- **Tổng số edge:** 1,942

## Nodes theo label

| Label | Số lượng |
|---|---:|
| `Clause` | 543 |
| `Point` | 359 |
| `Article` | 141 |
| `Obligation` | 72 |
| `Condition` | 47 |
| `Subject` | 45 |
| `Benefit` | 34 |
| `Organization` | 21 |
| `Section` | 13 |
| `LegalConcept` | 12 |
| `Chapter` | 11 |
| `ProhibitedAct` | 11 |
| `ExternalLaw` | 10 |
| `Right` | 7 |
| `Fund` | 6 |
| `Law` | 1 |
| `Role` | 1 |

## Edges theo type

| Type | Số lượng |
|---|---:|
| `HAS_CLAUSE` | 543 |
| `REFERENCES` | 387 |
| `HAS_POINT` | 359 |
| `HAS_ARTICLE` | 141 |
| `NEXT` | 140 |
| `IN_SECTION` | 110 |
| `ENTITLED_TO` | 82 |
| `HAS_OBLIGATION` | 70 |
| `CITES_EXTERNAL` | 30 |
| `HAS_SECTION` | 13 |
| `REQUIRES` | 13 |
| `DEFINES` | 12 |
| `HAS_CHAPTER` | 11 |
| `APPLIES_TO` | 7 |
| `RESPONSIBLE_FOR` | 7 |
| `MANAGES` | 6 |
| `REPLACES` | 5 |
| `AMENDS` | 2 |
| `PAID_FROM` | 2 |
| `REPEALS` | 1 |
| `HAS_RIGHT` | 1 |

## Top 10 Subjects theo độ degree

| Name | Degree | Xuất hiện trong (#Clause) |
|---|---:|---:|
| Người lao động | 61 | 103 |
| Người sử dụng lao động | 19 | 38 |
| Cơ quan bảo hiểm xã hội | 15 | 33 |
| Thân nhân | 10 | 21 |
| Lao động nữ | 5 | 22 |
| Hội đồng quản lý bảo hiểm xã hội | 4 | 11 |
| Công dân Việt Nam | 4 | 3 |
| Người tham gia bảo hiểm xã hội tự nguyện | 3 | 7 |
| Người thụ hưởng | 2 | 3 |
| Người tham gia | 2 | 5 |

## Top 10 Benefits

| Name | Category | Degree |
|---|---|---:|
| Lương hưu | `huu_tri` | 17 |
| Chế độ thai sản | `khac` | 16 |
| Hưu trí | `khac` | 13 |
| trợ cấp một lần | `khac` | 12 |
| Chế độ khi chăm sóc con ốm đau | `om_dau` | 8 |
| Bảo hiểm xã hội một lần | `khac` | 6 |
| trợ cấp mai táng | `khac` | 6 |
| Trợ cấp hưu trí xã hội | `khac` | 5 |
| Ốm đau | `khac` | 4 |
| Trợ cấp tuất hằng tháng | `khac` | 4 |

## External laws được viện dẫn

| Code | Title | #Cite |
|---|---|---:|
| `—` | Bộ luật Lao động | 15 |
| `84/2015/QH13` | Luật An toàn, vệ sinh lao động | 2 |
| `38/2013/QH13` | Luật | 2 |
| `39/2009/QH12` | Luật | 1 |
| `58/2014/QH13` | Luật Bảo hiểm xã hội | 5 |
| `35/2018/QH14` | Luật | 1 |
| `45/2019/QH14` | Bộ luật | 1 |
| `93/2015/QH13` | Nghị quyết | 1 |
| `01/2003/NĐ-CP` | Nghị định | 1 |
| `09/1998/NĐ-CP` | Nghị định | 1 |

## Article có ít/nhiều semantic edges nhất

**Top 10 Article có nhiều semantic edges nhất:**

| Article | #edges |
|---|---:|
| L41_2024.A18 — Trách nhiệm của cơ quan bảo hiểm xã hội | 13 |
| L41_2024.A3 — Giải thích từ ngữ | 12 |
| L41_2024.A111 — Chế độ hưu trí và chế độ tử tuất đối với người vừa có thời g | 5 |
| L41_2024.A119 — Sử dụng quỹ bảo hiểm xã hội | 5 |
| L41_2024.A15 — Quyền và trách nhiệm của tổ chức đại diện người sử dụng lao  | 5 |
| L41_2024.A58 — Trợ cấp một lần khi sinh con, nhận con khi nhờ mang thai hộ  | 5 |
| L41_2024.A94 — Đối tượng và điều kiện hưởng trợ cấp thai sản | 5 |
| L41_2024.A64 — Đối tượng và điều kiện hưởng lương hưu | 4 |
| L41_2024.A13 — Trách nhiệm của người sử dụng lao động | 4 |
| L41_2024.A75 — Tạm dừng, chấm dứt, tiếp tục hưởng lương hưu, trợ cấp bảo hi | 4 |

**Số Article không có semantic edge nào:** 48 / 141

<details><summary>Danh sách (cần review B3 prompt hoặc rerun)</summary>

- L41_2024.A1: Phạm vi điều chỉnh
- L41_2024.A103: Bảo lưu thời gian đóng bảo hiểm xã hội
- L41_2024.A105: Hồ sơ đề nghị hưởng lương hưu đối với người tham gia bảo hiểm xã hội tự nguyện
- L41_2024.A106: Hồ sơ đề nghị hưởng bảo hiểm xã hội một lần
- L41_2024.A108: Đối tượng hưởng chế độ tử tuất
- L41_2024.A11: Trách nhiệm của người tham gia và người thụ hưởng chế độ bảo hiểm xã hội
- L41_2024.A112: Hồ sơ đề nghị và giải quyết hưởng chế độ tử tuất
- L41_2024.A118: Các quỹ thành phần của quỹ bảo hiểm xã hội, quỹ bảo hiểm thất nghiệp
- L41_2024.A12: Quyền của người sử dụng lao động
- L41_2024.A120: Chi tổ chức và hoạt động bảo hiểm xã hội
- L41_2024.A121: Nguyên tắc đầu tư
- L41_2024.A122: Danh mục đầu tư và phương thức đầu tư
- L41_2024.A124: Đối tượng tham gia bảo hiểm hưu trí bổ sung
- L41_2024.A125: Nguyên tắc bảo hiểm hưu trí bổ sung
- L41_2024.A128: Quyền khiếu nại về bảo hiểm xã hội
- L41_2024.A129: Khiếu nại và giải quyết khiếu nại đối với quyết định hành chính, hành vi hành chính về bảo hiểm xã hội của cơ quan hành chính nhà nước, cơ quan bảo hiểm xã hội và người có thẩm quyền trong cơ quan hành chính nhà nước, cơ quan bảo hiểm xã hội
- L41_2024.A133: Nội dung quản lý nhà nước về bảo hiểm xã hội
- L41_2024.A134: Trách nhiệm quản lý nhà nước về bảo hiểm xã hội
- L41_2024.A135: Trách nhiệm của Chính phủ
- L41_2024.A136: Trách nhiệm của Bộ Lao động - Thương binh và Xã hội
- L41_2024.A137: Trách nhiệm của Bộ Tài chính
- L41_2024.A138: Trách nhiệm của Ủy ban nhân dân các cấp
- L41_2024.A14: Quyền và trách nhiệm của công đoàn, Mặt trận Tổ quốc Việt Nam và các tổ chức thành viên của Mặt trận
- L41_2024.A140: Hiệu lực thi hành
- L41_2024.A141: Quy định chuyển tiếp
- L41_2024.A17: Quyền hạn của cơ quan bảo hiểm xã hội
- L41_2024.A2: Đối tượng tham gia bảo hiểm xã hội bắt buộc và bảo hiểm xã hội tự nguyện
- L41_2024.A24: Trình tự, thủ tục thực hiện chế độ đối với người lao động không đủ điều kiện hưởng lương hưu và chưa đủ tuổi hưởng trợ cấp hưu trí xã hội
- L41_2024.A25: Sổ bảo hiểm xã hội
- L41_2024.A27: Hồ sơ đăng ký tham gia bảo hiểm xã hội bắt buộc và bảo hiểm xã hội tự nguyện
- L41_2024.A29: Điều chỉnh thông tin đăng ký kê khai tham gia bảo hiểm xã hội
- L41_2024.A30: Xác định đối tượng tham gia bảo hiểm xã hội bắt buộc và phát triển đối tượng tham gia bảo hiểm xã hội tự nguyện
- L41_2024.A32: Tỷ lệ đóng bảo hiểm xã hội
- L41_2024.A39: Trốn đóng bảo hiểm xã hội bắt buộc, bảo hiểm thất nghiệp
- L41_2024.A46: Dưỡng sức, phục hồi sức khoẻ sau khi ốm đau
- L41_2024.A5: Nguyên tắc bảo hiểm xã hội
- L41_2024.A67: Điều chỉnh lương hưu
- L41_2024.A7: Mức tham chiếu
- L41_2024.A71: Bảo lưu thời gian đóng bảo hiểm xã hội
- L41_2024.A74: Thực hiện bảo hiểm xã hội khi áp dụng chế độ tiền lương theo vị trí việc làm, chức danh và chức vụ lãnh đạo thay thế cho hệ thống bảng lương hiện hành
- L41_2024.A80: Hồ sơ đề nghị tiếp tục hưởng lương hưu, trợ cấp bảo hiểm xã hội hằng tháng trong trường hợp đã bị tạm dừng hoặc chấm dứt hưởng
- L41_2024.A83: Hồ sơ, trình tự khám giám định mức suy giảm khả năng lao động để giải quyết chế độ bảo hiểm xã hội
- L41_2024.A84: Đối tượng hưởng chế độ tử tuất
- L41_2024.A9: Các hành vi bị nghiêm cấm
- L41_2024.A93: Hình thức chi trả lương hưu và chế độ bảo hiểm xã hội
- L41_2024.A96: Hồ sơ đề nghị hưởng trợ cấp thai sản
- L41_2024.A97: Giải quyết hưởng trợ cấp thai sản
- L41_2024.A98: Đối tượng và điều kiện hưởng lương hưu

</details>