# PYNQ-Z2 Quality Inspection System: คู่มือการใช้งานและการติดตั้ง (YOLOv8n + Decision Tree ML)

โปรเจกต์นี้เป็นระบบตรวจสอบคุณภาพแผ่นอะลูมิเนียมกึ่งอัตโนมัติ โดยใช้สถาปัตยกรรมไฮบริดที่บูรณาการ **Deep Learning (YOLOv8n) สำหรับตรวจจับวัตถุ** และ **Machine Learning (Decision Tree) สำหรับตัดสินใจและประเมินผล QC** พร้อมระบบควบคุมไฟ LED/Buzzer แจ้งเตือนบนบอร์ด PYNQ-Z2

---

## 📂 รายการไฟล์ในโปรเจกต์ (`C:\Users\Anant\Desktop\เจค`)

* 📄 **`main_app.py`:** สคริปต์หลักรันเว็บเซิร์ฟเวอร์ Flask ดึงภาพจากกล้อง แปลงคุณลักษณะ (7 มิติ) ส่งให้ ML และบันทึกประวัติการตรวจสอบลงฐานข้อมูล SQLite
* 📄 **`yolo_detector.py`:** ตัวจัดการโมเดลตรวจจับวัตถุ YOLOv8n (สำหรับดึงรอยตำหนิและรูเจาะ M3-M6)
* 📄 **`ml_classifier.py`:** ตัวประเมินผล QC โดยรันกฎโมเดล Decision Tree ด้วยภาษา Python มาตรฐาน (เบาและไม่ต้องลง sklearn บนบอร์ด)
* 📄 **`train_ml.py`:** สคริปต์ฝึกสอนโมเดล Decision Tree บน PC เพื่อส่งออกเป็นไฟล์กฎเกณฑ์รูปแบบ JSON
* 📄 **`db_manager.py`:** ตัวจัดการและสร้างฐานข้อมูล SQLite (`quality_inspection.db`) บันทึกประวัติและดึงสถิติ
* 📄 **`hardware_control.py`:** ตัวส่งเอาต์พุตคุมสัญญาณไฟ LED และเสียงเตือนของ Buzzer ผ่าน GPIO ของบอร์ด (มี Mock mode บน Windows)
* 📁 **`templates/`**
  * 📄 **`index.html`** - หน้าจอควบคุม Web Dashboard ล้ำสมัย แสดงผลภาพกล้อง, กล่องฟีเจอร์ ML, และตารางประวัติ Real-time
* 📁 **`models/`**
  * 📄 **`ml_model.json`** - ไฟล์ JSON บันทึกกฎการประเมินผลของแมชชีนเลิร์นนิง
  * 📄 **`.gitkeep`** - โฟลเดอร์สำหรับอัปโหลดโมเดล YOLO (`yolov8n.onnx` หรือ `yolov8n.pt`)

---

## 💻 1. การรันเพื่อจำลองการทำงานบนคอมพิวเตอร์ส่วนบุคคล (Local Simulation - Windows)

คุณสามารถทดสอบซอฟต์แวร์ ฐานข้อมูล SQLite และหน้า Dashboard ทั้งหมดบน Windows ได้โดยไม่ต้องต่อบอร์ด PYNQ:

1. ติดตั้งไลบรารีที่จำเป็นบน Windows (ผ่าน Command Prompt / PowerShell):
   ```bash
   pip install flask opencv-python numpy
   ```
2. รันแอปพลิเคชันหลัก:
   ```bash
   cd C:\Users\Anant\Desktop\เจค
   python main_app.py
   ```
3. เปิดเว็บเบราว์เซอร์ไปที่:
   ```
   http://localhost:5000
   ```
4. **การทดสอบจำลอง (Mock Mode):**
   * กดปุ่ม **"ตรวจสอบชิ้นงาน (Inspect)"** ระบบจะส่งฟีเจอร์ไปให้โมเดล ML ประเมินผลผ่าน PASS
   * หากเปิดสวิตช์ **"จำลองรอยขีดข่วนชำรุด"** แล้วกด Inspect อีกครั้ง ตัวเลขอินพุตจะถูกส่งไปประมวลผลผ่านโมเดลต้นไม้ตัดสินใจ และแสดงผลลัพธ์เป็น FAIL พร้อมระบุสาเหตุแบบ Real-time บนตารางแดชบอร์ด

---

## ⚡ 2. การเตรียมไฟล์โมเดล YOLOv8n และดัชนีคลาส

ดัชนีคลาส (Class Index Mapping) ของโมเดล YOLOv8n ถูกกำหนดไว้ดังนี้ในไฟล์ `yolo_detector.py`:
* `0` = `scratch` (รอยขีดข่วนชำรุด)
* `1` = `M3`
* `2` = `M4`
* `3` = `M5`
* `4` = `M6`

### การแปลงโมเดล PyTorch (.pt) เป็น ONNX:
รันใน Python บนคอมพิวเตอร์หลักของคุณ:
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt") 
model.export(format="onnx", imgsz=640)
```
หลังจากได้ไฟล์ `yolov8n.onnx` แล้ว ให้นำไปวางไว้ในโฟลเดอร์ `models` ของโปรเจกต์ที่:
`C:\Users\Anant\Desktop\เจค\models\yolov8n.onnx`

---

## 🔌 3. การใช้งานบนบอร์ด PYNQ-Z2 จริง

1. เชื่อมต่อสาย **USB Webcam** เข้ากับพอร์ต USB Host ของบอร์ด PYNQ-Z2
2. ต่อสายสัญญาณของ **Buzzer** เข้าที่พอร์ต PMOD A Pin 1 (ขาบวก) และ GND (ขาลบ)
3. โอนย้ายโฟลเดอร์ `เจค` ทั้งหมดไปไว้บนบอร์ด PYNQ-Z2 (ผ่าน Jupyter Notebook Dashboard ที่พอร์ต `9090` หรือใช้ WinSCP/FileZilla เชื่อมต่อ SSH เข้าทางพอร์ต `22` ด้วย Username: `xilinx` / Password: `xilinx` ที่โฟลเดอร์ `/home/xilinx/`)
4. เชื่อมต่อบอร์ดด้วยโปรแกรม **Tera Term** ตั้งค่า Baud rate `115200`
5. ไปที่โฟลเดอร์โปรเจกต์บนบอร์ดและรันแอปพลิเคชัน (จำเป็นต้องใช้ sudo เพื่อควบคุมฮาร์ดแวร์):
   ```bash
   cd /home/xilinx/เจค
   sudo pip3 install flask
   sudo python3 main_app.py
   ```
6. เข้าแดชบอร์ดควบคุมโดยใช้ IP ของบอร์ด เช่น:
   ```
   http://<IP ของบอร์ด PYNQ-Z2>:5000
   ```
   คุณสามารถสตรีมกล้องสดเพื่อตรวจสอบชิ้นงานแผ่นอะลูมิเนียมจริง และสั่งควบคุม Buzzer/LED บนบอร์ด PYNQ-Z2 ได้โดยตรง!
