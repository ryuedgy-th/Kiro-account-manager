# Web Admin Dashboard — คู่มือใช้งาน & ตั้ง Cloudflare Access

## สิ่งที่เพิ่มเข้าไปในโค้ด

หน้า **`/admin`** = web dashboard จัดการระบบ (self-contained HTML เหมือน `/portal`) เสิร์ฟจาก proxy server ตัวเดิม port เดิม แยกด้วย path

**ความสามารถ:**
- 📊 **ภาพรวม** — requests/tokens/credits/uptime, สุขภาพ account pool, คำขอล่าสุด
- 👥 **ลูกค้า** — สร้าง / เติม-หัก credit (มี note) / enable-disable / reset password / ลบ
- 🔑 **API Keys (operator)** — สร้าง (sk/simple/token) / ตั้ง credit limit / enable-disable / reset usage / ลบ
- ✉️ **Invites** — สร้าง invite code / ดู / ยกเลิก
- 📜 **Audit** — บันทึกการกระทำล่าสุด + ปุ่มล้าง cache

## ความปลอดภัย — 2 ชั้น
```
[Internet] → Cloudflare Access (ชั้น 1: login Google/OTP) → Tunnel → /admin → operator key (ชั้น 2)
```
- หน้า `/admin` แสดงเฉพาะเมื่อ `adminApiExposed = true` (ปิด = 404 ไม่บอกว่ามี route นี้)
- หน้า HTML โหลดได้โดยไม่ต้องมี key แต่ **ทุก API `/admin/*` ต้องมี operator key** (customer key ใช้ไม่ได้ — กันการ提权)
- CORS ของ `/admin` ไม่ส่ง wildcard origin (เว็บอื่นอ่าน response ไม่ได้)

---

## ขั้นตอนเปิดใช้

### 1. เปิด admin API + ตั้ง operator key (ทำผ่าน GUI Electron ครั้งเดียว)
- เปิด `adminApiExposed = true`
- มี API key อย่างน้อย 1 ตัวที่**ไม่ผูกกับลูกค้า** (operator key) — ใช้อันนี้ login หน้าเว็บ
- เพิ่ม origin ของ admin เข้า `portalAllowedOrigins` เช่น `https://admin.yourdomain.com`

### 2. Cloudflare Tunnel — route path
ใน `~/.cloudflared/config.yml` (หรือ dashboard):

**แบบ A — แยก subdomain (แนะนำ):**
```yaml
ingress:
  - hostname: admin.yourdomain.com
    service: http://localhost:<proxy_port>
  - hostname: portal.yourdomain.com
    service: http://localhost:<proxy_port>
  - service: http_status:404
```
> หมายเหตุ: ทั้งสอง subdomain ชี้ service เดียวกัน เพราะ path `/admin` vs `/portal` แยกเองในแอป ผู้ใช้เข้า `admin.yourdomain.com/admin`

**แบบ B — โดเมนเดียว:** ชี้ `app.yourdomain.com` → service เดียว แล้วเข้า `app.yourdomain.com/admin`

### 3. Cloudflare Zero Trust — Access Application (หัวใจความปลอดภัย)
1. ไป **Zero Trust dashboard → Access → Applications → Add an application → Self-hosted**
2. ตั้ง Application:
   - **แบบ A:** Subdomain = `admin`, Domain = `yourdomain.com`, Path = เว้นว่าง (ครอบทั้ง subdomain)
   - **แบบ B:** Subdomain = `app`, Path = `admin` (ครอบเฉพาะ `/admin*` — ที่เหลือเปิดปกติ)
3. **Policies → Add a policy:**
   - Action: **Allow**
   - Include: **Emails** = อีเมลคุณ (หรือ Email domain ของบริษัท)
4. Save

> ผลลัพธ์: ใครเข้า `/admin` จะเจอหน้า login Google/OTP ของ Cloudflare **ก่อน** ถึงแอป — bot ที่ scan เจอจะโดนกั้นตั้งแต่ขอบ ส่วน `/portal` กับ `/v1/*` ลูกค้าเข้าได้ปกติไม่ต้อง login Cloudflare

### 4. เข้าใช้งาน
เปิด `https://admin.yourdomain.com/admin` → ผ่าน Cloudflare login → หน้า admin → ใส่ operator key → ใช้งาน

---

## ⚠️ ข้อควรระวัง
- **อย่าเปิด `adminApiExposed = true` โดยไม่มี Cloudflare Access** — ถ้าไม่มี ชั้นกันจะเหลือแค่ operator key ด่านเดียว
- operator key เก็บใน localStorage ของ browser (เหมือน portal เก็บ token) — ใช้เครื่องส่วนตัว
- ฟังก์ชันที่**ไม่ได้**เอาขึ้นเว็บ (ตั้งใจ เพื่อความปลอดภัย): start/stop/restart proxy, แก้ config, จัดการ account pool, machine ID, ติดตั้ง CA, login เพิ่มบัญชี — งานพวกนี้ทำผ่าน Electron GUI เท่านั้น

## Follow-up ที่ยังไม่ได้ทำ (ถ้าต้องการเพิ่มภายหลัง)
- rate-limit เฉพาะ admin (ตอนนี้พึ่ง Cloudflare Access เป็นชั้นกันหลัก)
- ดู usage history รายลูกค้าบนหน้าเว็บ (ตอนนี้มีแต่ยอดรวม — รายละเอียดดูที่ portal ของลูกค้า)
