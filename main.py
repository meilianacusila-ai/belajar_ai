import mysql.connector
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

# Load environment variable
load_dotenv()

# Ambil kredensial MySQL
user_name = os.environ.get("MYSQL_USERNAME")
password = os.environ.get("MYSQL_PASSWORD")
database = os.environ.get("DB_NAME")

try:
    # Koneksi ke MySQL
    conn = mysql.connector.connect(
        host="localhost",
        user=user_name,
        password=password,
        database=database,
    )

    if conn.is_connected():
        print("Successfully connecting to MySQL database!")
        cursor = conn.cursor()

        # ======================================================
        #                FUNGSI–FUNGSI PROGRAM
        # ======================================================

        # 1. Daftar harga kamar
        def harga_kamar(cursor):
            cursor.execute("SELECT * FROM harga h")
            results = cursor.fetchall()
            print("\n=== Daftar Harga Kamar ===")
            print("ID_Kelas  | Kamar | Harga                | Jmlh_Tersedia")
            for row in results:
                print(f" {row[0]:<8} | {row[1]:<5} | {row[2]:<20} | {row[3]:<10}")

        # 2. Daftar rawat inap
        def daftar_rawat_inap(cursor):
            cursor.execute("SELECT * FROM inap")
            results = cursor.fetchall()

            def safe(v): return v if v is not None else ""

            print("\n=== Daftar Rawat Inap ===")
            print("Nama            | NIK            | Lahir     | Masuk      | Penjamin | Kls | Keluar")
            for row in results:
                print(
                    f"{safe(row[1]):<15} | "
                    f"{safe(row[2]):<14} | "
                    f"{safe(row[3]):<10} | "
                    f"{safe(row[4]):<10} | "
                    f"{safe(row[5]):<8} | "
                    f"{safe(row[6]):<3} | "
                    f"{safe(row[7])}"
                )

        # 3. Tambah pasien
        def tambah_pasien(cursor):
            Nama = input("Nama pasien        : ")
            NIK = input("NIK                : ")
            Tgl_Lahir = input("Tanggal lahir (YYYY-MM-DD): ")
            ID_Kelas = int(input("ID Kelas           : "))
            Penjamin = input("Penjamin           : ")
            Tgl_Masuk = input("Tanggal masuk (YYYY-MM-DD): ")

            query = """
                INSERT INTO inap (Nama, NIK, Tgl_Lahir, ID_Kelas, Penjamin, Tgl_Masuk)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (Nama, NIK, Tgl_Lahir, ID_Kelas, Penjamin, Tgl_Masuk))
            conn.commit()
            print("✅ Data pasien berhasil ditambahkan!")

        # 4. Update tanggal keluar
        def update_tanggal_keluar(cursor, conn):
            Nama = input("Nama pasien      : ")
            Tgl_Lahir = input("Tanggal Lahir (YYYY-MM-DD): ")
            Tgl_Keluar = input("Tanggal Keluar (YYYY-MM-DD): ")

            query_update = """
                UPDATE inap
                SET Tgl_Keluar = %s
                WHERE Nama LIKE %s AND Tgl_Lahir = %s
            """
            cursor.execute(query_update, (Tgl_Keluar, f"%{Nama}%", Tgl_Lahir))
            conn.commit()

            if cursor.rowcount > 0:
                cursor.execute("""
                    UPDATE inap
                    SET Lama_Inap = DATEDIFF(Tgl_Keluar, Tgl_Masuk)
                    WHERE Nama LIKE %s AND Tgl_Lahir = %s
                """, (f"%{Nama}%", Tgl_Lahir))
                conn.commit()

                print("✅ Tanggal keluar diperbarui!")
                print("📌 Lama inap dihitung otomatis!")
            else:
                print("❌ Tidak ada data yang cocok.")

        # 5. Total harga rawat inap
        def total_harga(cursor):
            print("\n=== Cetak Total Rawat Inap ===")
            Nama = input("Masukkan Nama Pasien     : ")
            Tgl_Lahir = input("Masukkan Tanggal Lahir   : ")

            cursor.execute("""
                SELECT 
                    i.Nama, i.ID_Kelas,
                    DATEDIFF(i.Tgl_Keluar, i.Tgl_Masuk) AS Lama_Inap,
                    h.Harga_per_hari
                FROM inap i
                JOIN harga h ON i.ID_Kelas = h.ID_Kelas
                WHERE i.Nama = %s AND i.Tgl_Lahir = %s
            """, (Nama, Tgl_Lahir))

            row = cursor.fetchone()
            if not row:
                print("❌ Data tidak ditemukan.")
                return

            nama, kelas, lama, harga = row

            total = lama * harga
            print("\n=== Total Biaya Rawat Inap ===")
            print(f"Nama Pasien      : {nama}")
            print(f"Kelas Kamar      : {kelas}")
            print(f"Lama Inap        : {lama} hari")
            print(f"Harga per Hari   : Rp {harga:,}")
            print(f"Total Biaya      : Rp {total:,}")
            print("=============================\n")

        # 6. Diagram batang rata-rata lama inap
        def diagram_batang(cursor):
            cursor.execute("""
                SELECT ID_Kelas, AVG(Lama_Inap)
                FROM inap
                WHERE Lama_Inap IS NOT NULL
                GROUP BY ID_Kelas
                ORDER BY ID_Kelas
            """)
            data = cursor.fetchall()

            kelas = [str(d[0]) for d in data]
            rata2 = [int(d[1]) for d in data]

            plt.figure(figsize=(8, 5))
            plt.bar(kelas, rata2)
            plt.title("Rata-rata Lama Inap per Kelas")
            plt.xlabel("ID Kelas")
            plt.ylabel("Rata-rata Hari")
            plt.grid(axis='y', linestyle='--')
            plt.show()

        # 7. Statistik Dasar (deskriptif)
        def statistik_dasar(cursor):
            print("\n=== Statistik Dasar Rawat Inap ===")

            cursor.execute("""
                SELECT COUNT(*), AVG(Lama_Inap), MIN(Lama_Inap), MAX(Lama_Inap)
                FROM inap
                WHERE Lama_Inap IS NOT NULL
            """)

            total, rata2, minimum, maksimum = cursor.fetchone()
            rata2 = int(rata2) if rata2 else 0

            print(f"Total pasien               : {total}")
            print(f"Rata-rata lama inap        : {rata2} hari")
            print(f"Lama inap minimum          : {minimum} hari")
            print(f"Lama inap maksimum         : {maksimum} hari")

            print("\n--- Statistik Per Kelas ---")
            cursor.execute("""
                SELECT ID_Kelas, COUNT(*), AVG(Lama_Inap), MIN(Lama_Inap), MAX(Lama_Inap)
                FROM inap
                WHERE Lama_Inap IS NOT NULL
                GROUP BY ID_Kelas
                ORDER BY ID_Kelas
            """)

            rows = cursor.fetchall()
            for r in rows:
                kelas, tot, avg, mn, mx = r
                print(f"\nKelas {kelas}:")
                print(f"   Jumlah pasien          : {tot}")
                print(f"   Rata-rata lama inap    : {int(avg) if avg else 0} hari")
                print(f"   Minimal lama inap      : {mn}")
                print(f"   Maksimal lama inap     : {mx}")

        # ======================================================
        #                      MENU
        # ======================================================

        def menu():
            print("=====================================")
            print("   Selamat Datang di RS SUCI 💙")
            print("=====================================")
            print("""
    1. Daftar Harga Kamar
    2. Daftar Pasien Rawat Inap
    3. Tambah Data Pasien Baru
    4. Update Tanggal Keluar Pasien
    5. Total Biaya Rawat Inap
    6. Diagram Statistik
    7. Statistik Deskriptif
    8. Keluar
            """)
            return input("Masukkan pilihan (1-8): ")

        # ======================================================
        #                      MAIN LOOP
        # ======================================================

        def main():
            while True:
                pil = menu()

                if pil == "1":
                    harga_kamar(cursor)
                elif pil == "2":
                    daftar_rawat_inap(cursor)
                elif pil == "3":
                    tambah_pasien(cursor)
                elif pil == "4":
                    update_tanggal_keluar(cursor, conn)
                elif pil == "5":
                    total_harga(cursor)
                elif pil == "6":
                    diagram_batang(cursor)
                elif pil == "7":
                    statistik_dasar(cursor)
                elif pil == "8":
                    print("Terima kasih telah menggunakan sistem RS SUCI 🙏")
                    break
                else:
                    print("Pilihan tidak valid!")

        # ============================
        #          JALANKAN
        # ============================
        main()

except mysql.connector.Error as err:
    print(f"Error: {err}")

finally:
    try:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals() and conn.is_connected():
            conn.close()
            print("Connection closed!")
    except Exception as e:
        print(f"Gagal menutup koneksi: {e}")
