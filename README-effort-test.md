# Maxplus Effort Parameter Test

สคริปต์สำหรับทดสอบว่า `effort` parameter มีผลจริงต่อการทำงานของ Maxplus API หรือไม่

## การใช้งาน

### วิธีที่ 1: ใช้ Maxplus proxy ภายในโปรเจคนี้ (แนะนำ)

```bash
# เริ่ม Maxplus proxy ของคุณก่อน
cd Kiro-account-manager
npm run dev

# จากนั้นรันสคริปต์ทดสอบ (terminal อื่น)
npx tsx test-effort-impact-simple.ts http://localhost:3000 logic
```

### วิธีที่ 2: ทดสอบกับ Maxplus endpoint โดยตรง

```bash
# ตั้งค่า environment variable
export MAXPLUS_BASE_URL=https://your-maxplus-endpoint.com

# รันทดสอบ
npx tsx test-effort-impact-simple.ts https://your-maxplus-endpoint.com logic
```

## ประเภทโจทย์ทดสอบ

1. **logic** - ปริศนาข้ามแม่น้ำ (ต้องใช้ reasoning หลายขั้นตอน)
2. **math** - โจทย์สมการ cubic (ต้องคิดวิเคราะห์)
3. **code** - หา bug ใน binary search (ต้องใช้ logic debugging)

## ผลลัพธ์ที่ได้

สคริปต์จะทดสอบ 4 ระดับ: `low`, `medium`, `high`, `max`

### สัญญาณที่บอกว่า effort มีผล:
- ✅ Output tokens เพิ่มขึ้น > 10%
- ✅ Response time เพิ่มขึ้น > 20%
- ✅ Thinking content เพิ่มขึ้น

### สัญญาณที่บอกว่า effort ไม่มีผล:
- ❌ Token usage คงที่ทุก effort
- ❌ Response time ไม่เปลี่ยน
- ❌ คุณภาพคำตอบเหมือนกันหมด

## ตัวอย่างผลลัพธ์

```
📊 Results Summary:
┌──────────┬─────────────┬──────────────┬────────────┬──────────────┐
│  Effort  │ Input Tokens│ Output Tokens│ Time (ms)  │ Thinking (ch)│
├──────────┼─────────────┼──────────────┼────────────┼──────────────┤
│ low      │         245 │          156 │       2340 │          324 │
│ medium   │         245 │          298 │       4120 │          856 │
│ high     │         245 │          445 │       6780 │         1542 │
│ max      │         245 │          623 │       9230 │         2234 │
└──────────┴─────────────┴──────────────┴────────────┴──────────────┘

📈 Variation Analysis:
   Output Tokens: 156 → 623 (+299.4%)
   Response Time: 2340ms → 9230ms (+294.4%)
   Thinking Length: 324 → 2234 chars (+589.5%)

🎯 Verdict:
   ✅ effort parameter HAS REAL IMPACT
      - Output tokens increase with effort
      - Response time increases with effort
      - Thinking content increases with effort
```

## หมายเหตุ

- สคริปต์จะรอ 2 วินาทีระหว่างแต่ละ request เพื่อหลีกเลี่ยง rate limiting
- โจทย์ทดสอบต้องการ reasoning จริงๆ เพื่อให้เห็นผลชัดเจน
- หาก Maxplus เป็นเพียง passthrough ที่ไม่ได้ประมวลผล effort จริง ตัวเลขจะคงที่
