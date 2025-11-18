import mysql.connector
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

# Load semua environment variable dari file .env
load_dotenv()

# Definisikan kredensial dari environment
user_name = os.environ.get("MYSQL_USERNAME")
password = os.environ.get("MYSQL_PASSWORD")
database = os.environ.get("DB_NAME")

try:
    # Membuat koneksi ke MySQL server
    conn = mysql.connector.connect(
        host="localhost",
        user=user_name,
        password=password,
        database=database,
    )

    # Apabila koneksi berhasil
    if conn.is_connected():
        print("Successfully connecting to MySQL database!")

        # Membuat object cursor untuk eksekusi SQL
        cursor = conn.cursor()

        # ==============================
        #   FUNGSI-FUNGSI PROGRAM
        # ==============================

        # 1. Tampilkan harga kamar
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

            def safe(val):
                return val if val is not None else ""

            print("\n=== Daftar Rawat Inap ===")
            print("Nama                                        | NIK              | Tgl_Lahir     | Tgl_Masuk | Penjamin   | ID_Kelas | Tgl_Keluar")
            for row in results:
                print(
                    f"{safe(row[1]):<18}\t|\t"
                    f"{safe(row[2]):<16}\t|\t"
                    f"{safe(row[3]):<15}\t|\t"
                    f"{safe(row[4]):<15}\t|\t"
                    f"{safe(row[5]):<12}\t|\t"
                    f"{safe(row[6]):<8}\t|\t"
                    f"{safe(row[7]):<12}"
                )

        # 3. Tambah pasien masuk
        def tambah_pasien(cursor):
            Nama = input("Nama pasien        : ")
            NIK = input("NIK                : ")
            Tgl_Lahir = input("Tanggal lahir (YYYY-MM-DD): ")
            ID_Kelas = int(input("ID Kelas           : "))
            Penjamin = input("Penjamin : ")
            Tgl_Masuk = input("Tanggal masuk (YYYY-MM-DD): ")

            query = (
                "INSERT INTO inap (Nama, NIK, Tgl_Lahir, ID_Kelas, Penjamin, Tgl_Masuk) "
                "VALUES (%s, %s, %s, %s, %s, %s);"
            )
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
                query_lama = """
                    UPDATE inap
                    SET Lama_Inap = DATEDIFF(Tgl_Keluar, Tgl_Masuk)
                    WHERE Nama LIKE %s AND Tgl_Lahir = %s
                """
                cursor.execute(query_lama, (f"%{Nama}%", Tgl_Lahir))
                conn.commit()

                print("✅ Tanggal keluar diperbarui!")
                print("📌 Lama inap dihitung otomatis!")
            else:
                print("❌ Tidak ada data yang cocok. Periksa Nama & Tanggal Lahir.")

        # 5. Total biaya rawat inap
        def total_harga(cursor):
            print("\n=== Cetak Total Rawat Inap ===")
            Nama = input("Masukkan Nama Pasien (tepat sama)     : ")
            Tgl_Lahir = input("Masukkan Tanggal Lahir (YYYY-MM-DD): ")

            cursor.execute("""
                SELECT 
                    i.Nama,
                    i.ID_Kelas,
                    DATEDIFF(i.Tgl_Keluar, i.Tgl_Masuk) AS Lama_Inap,
                    h.Harga_per_hari
                FROM inap i
                JOIN harga h ON i.ID_Kelas = h.ID_Kelas
                WHERE i.Nama = %s AND i.Tgl_Lahir = %s;
            """, (Nama, Tgl_Lahir))

            row = cursor.fetchone()

            if row is None:
                print("\n❌ Data pasien tidak ditemukan.")
                return

            nama, kelas, lama_inap, harga_per_hari = row

            if lama_inap is None:
                print("❌ Lama inap tidak dapat dihitung. Pastikan Tgl_Masuk dan Tgl_Keluar terisi.")
                return

            if harga_per_hari is None:
                print("❌ Harga per hari tidak ditemukan untuk ID_Kelas tersebut.")
                return

            total_biaya = lama_inap * harga_per_hari

            print("\n=== Total Biaya Rawat Inap ===")
            print(f"Nama Pasien      : {nama}")
            print(f"Kelas Kamar      : {kelas}")
            print(f"Lama Inap        : {lama_inap} hari")
            print(f"Harga per Hari   : Rp {harga_per_hari:,}")
            print(f"Total Biaya      : Rp {total_biaya:,}")
            print("=============================\n")

        # 6. Statistik diagram batang
        def diagram_batang(cursor):
            cursor.execute("""
                SELECT ID_Kelas, AVG(Lama_Inap)
                FROM inap
                WHERE Lama_Inap IS NOT NULL
                GROUP BY ID_Kelas
                ORDER BY ID_Kelas ASC;
            """)

            data = cursor.fetchall()
            kelas = [str(row[0]) for row in data]
            rata2_lama = [row[1] for row in data]

            plt.figure(figsize=(8, 5))
            plt.bar(kelas, rata2_lama)
            plt.title("Rata-rata Lama Inap per Kelas Kamar")
            plt.xlabel("ID Kelas")
            plt.ylabel("Rata-rata Lama Inap (hari)")
            plt.grid(axis='y', linestyle='--', linewidth=0.5)
            plt.show()

        # 7. Menu utama
        def menu():
            print("=====================================")
            print("   Selamat Datang di RS SUCI 💙")
            print("=====================================")

            print("""
        Menu Utama:
        1. Daftar Harga Kamar
        2. Daftar Pasien Rawat Inap
        3. Tambah Data Pasien Rawat Inap Baru
        4. Update Tanggal Keluar Pasien
        5. Total Biaya
        6. Statistik 
        7. Keluar
            """)

            return input("Masukkan pilihan (1-7): ")

        # 8. Main loop
        def main():
            run = True
            while run:
                pilihan = menu()
                if pilihan == "1":
                    harga_kamar(cursor)
                elif pilihan == "2":
                    daftar_rawat_inap(cursor)
                elif pilihan == "3":
                    tambah_pasien(cursor)
                elif pilihan == "4":
                    update_tanggal_keluar(cursor, conn)
                elif pilihan == "5":
                    total_harga(cursor)
                elif pilihan == "6":
                    diagram_batang(cursor)
                elif pilihan == "7":
                    print("Terima kasih telah mengunjungi RS SUCI 🙏")
                    run = False
                else:
                    print("Pilihan tidak valid, coba lagi.")


        # ===============================
        # 🔥 PANGGIL MAIN DI DALAM TRY 🔥
        # ===============================
        main()

except mysql.connector.Error as err:
    print(f"Error: {err}")

finally:
    try:
        if 'cursor' in locals() and cursor is not None:
            cursor.close()

        if 'conn' in locals() and conn is not None and conn.is_connected():
            conn.close()
            print("Connection closed!")

    except Exception as e:
        print(f"Gagal menutup koneksi: {e}")