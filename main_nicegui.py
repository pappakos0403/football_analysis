from nicegui import ui, events, storage, app
from pathlib import Path

# ----------------
# --- VÁLTOZÓK ---
# ----------------

# Feltöltött videók könyvtára
INPUT_DIR = Path("input_videos")
INPUT_DIR.mkdir(exist_ok=True)

# Kiválasztott videó tárolása
selected_video = None

# ----------------
# --- FUNKCIÓK ---
# ----------------

# Videó feltöltése és mentése a megadott könyvtárba
def save_uploaded_video(e: events.UploadEventArguments):
    target_path = INPUT_DIR / e.name
    # Videó átmásolása a megadott könyvtárba
    with open(target_path, 'wb') as f:
        f.write(e.content.read())
    ui.notify(f"Videó feltöltve: {e.name}")

# Adott videófájl törlése gombnyomásra
def delete_video_file(file_path: Path):
    if file_path.exists():
        file_path.unlink()
        ui.notify(f"Törölve: {file_path.name}")
        # Oldal frissítése a törlés után
        ui.navigate.to('/uploaded_videos')

# Videó kiválasztása
def select_video_for_analysis(video_file: Path):
    global selected_video
    selected_video = video_file
    ui.notify(f"Kiválasztott videó: {video_file.name}")
    ui.navigate.to('/analysis_config')

# ---------------
# --- OLDALAK ---
# ---------------

# --- Kezdőlap ("/") ---
@ui.page("/")
def start_page():
    with ui.column().classes("absolute-center items-center gap-12"):
        # Kezdőlap címe
        ui.label("Futballanalízis Alkalmazás").classes("text-3xl font-bold")
        # Indítás gomb -> teszt oldalra navigál
        ui.button("Indítás", on_click=lambda: ui.navigate.to("/main_page")).classes("mt-4")
        # Kilépés gomb -> alkalmazás leállítása
        ui.button("Kilépés", on_click=app.shutdown).classes("mt-4")

# --- Feltöltött videók oldala ("/uploaded_videos") ---
@ui.page("/uploaded_videos")
def uploaded_videos_page():
    with ui.column().classes("absolute-center items-center gap-4"):
        # Feltöltött videók oldalának címe
        ui.label("Feltöltött videók").classes("text-2xl font-semibold")
        # Feltöltött videók mappájának ellenőrzése
        video_files = list(INPUT_DIR.glob("*.mp4")) + list(INPUT_DIR.glob("*.avi")) + list(INPUT_DIR.glob("*.mkv"))
        if not video_files:
            ui.label("A Feltöltött videók mappája üres!").classes("text-white")
        else:
            with ui.row().classes("justify-center items-start flex-wrap gap-8"):
                for video_file in video_files:
                    with ui.column().classes("items-center"):
                        # Thumbnail + videólejátszó
                        ui.video(f'/videos/{video_file.name}').props('controls').classes("w-64 h-36 rounded shadow")
                        # Videók címei
                        ui.label(video_file.name).classes("text-sm text-center text-white mt-2")
                        # Videó törlése gomb
                        ui.button(on_click=lambda f=video_file: delete_video_file(f))\
                            .props('color="red" icon="delete"')
        # Visszalépés a menübe
        ui.button("Vissza a menübe", on_click=lambda: ui.navigate.to("/main_page")).classes("mt-4")

# --- Videó kiválasztása elemzéshez ("/select_video_for_analysis") ---
@ui.page("/select_video_for_analysis")
def select_video_for_analysis_page():
    with ui.column().classes("absolute-center items-center gap-4"):
        # Videó kiválasztása oldal címe
        ui.label("Videó kiválasztása elemzéshez").classes("text-2xl font-semibold")
        ui.markdown("Kattintson a videóra a kiválasztáshoz!").classes("text-center")
        
        # Feltöltött videók mappájának ellenőrzése
        video_files = list(INPUT_DIR.glob("*.mp4")) + list(INPUT_DIR.glob("*.avi")) + list(INPUT_DIR.glob("*.mkv"))
        if not video_files:
            ui.label("Nincs feltöltött videó! Először töltsön fel videókat.").classes("text-orange-400 text-center")
            ui.button("Videó feltöltése", on_click=lambda: ui.navigate.to("/main_page")).classes("mt-4")
        else:
            with ui.row().classes("justify-center items-start flex-wrap gap-8 max-w-4xl"):
                for video_file in video_files:
                    with ui.column().classes("items-center cursor-pointer"):
                        # Kattintható videó container
                        with ui.card().classes("p-4 hover:bg-gray-700 transition-colors duration-200") \
                            .on('click', lambda f=video_file: select_video_for_analysis(f)):
                            # Videó thumbnail (első frame előnézet) - letiltott vezérlőkkel
                            ui.video(f'/videos/{video_file.name}').classes("w-64 h-36 rounded shadow") \
                                .props('muted preload="metadata" style="pointer-events: none;"')
                            # Videók címei
                            ui.label(video_file.name).classes("text-sm text-center text-white mt-2 max-w-64 truncate")
        
        # Visszalépés az elemzés konfigurációs oldalra
        ui.button("Vissza", on_click=lambda: ui.navigate.to("/analysis_config")).classes("mt-4")

# --- Elemzés konfigurációs oldal ("/analysis_config") ---
@ui.page("/analysis_config")
def analysis_config_page():
    with ui.column().classes("absolute-center items-center gap-4"):
        # Elemzés konfigurációs oldal címe
        ui.label("Videó elemzése").classes("text-2xl font-semibold")
        
        # Kiválasztott videó megjelenítése
        if selected_video:
            ui.markdown(f"**Kiválasztott videó:** {selected_video.name}").classes("text-center text-green-400")
        else:
            ui.markdown("Még nincs videó kiválasztva az elemzéshez.").classes("text-center text-orange-400")
        
        # Videó kiválasztása gomb
        ui.button("Videó kiválasztása", 
                 on_click=lambda: ui.navigate.to("/select_video_for_analysis")).classes("mt-4")
        
        # Elemzés indítása gomb (csak ha van kiválasztott videó)
        if selected_video:
            ui.button("Elemzés indítása", 
                     on_click=lambda: ui.notify("Az elemzés funkció még fejlesztés alatt áll!")).classes("mt-4")
            
        # Vissza a menübe gomb -> visszalépés a főoldalra
        ui.button("Vissza a menübe", on_click=lambda: ui.navigate.to("/main_page")).classes("mt-4")

# --- Elemezni kívánt videó kiválasztása oldal ---
@ui.page("/select_video")
def select_video_page():
    with ui.column().classes("absolute-center items-center gap-4"):
        # Videó kiválasztása oldal címe
        ui.label("Videó kiválasztása").classes("text-2xl font-semibold")
        # Videó kiválasztása gomb --> videó kiválasztása oldalra navigál
        ui.button("Videó kiválasztása", on_click=lambda: ui.navigate.to("/analysis_config")).classes("mt-4")
        # Vissza az elemzés konfigurációs oldalra --> elemzés konfigurációs oldalra navigál
        ui.button("Vissza", on_click=lambda: ui.navigate.to("/main_page")).classes("mt-4")

# --- Teszt oldal ("/teszt") ---
@ui.page("/main_page")
def main_page():
    with ui.column().classes("absolute-center items-center"):
        # Teszt oldal címe
        ui.label("Videó elemzése").classes("text-2xl font-semibold")
        # Videó feltöltése
        ui.upload(on_upload=save_uploaded_video,
                  multiple=True, 
                  on_rejected=lambda e: ui.notify("Csak mp4/avi kiterjesztésű videók engedélyezettek!")
                  ).props('accept=".mp4,.avi,.mkv" auto-upload')  
        # Videó feltöltésének leírása
        ui.markdown("Itt töltheti fel az elemezni kívánt videót (mp4/avi/mkv).").classes("text-center")
        # Feltöltött videók -> feltöltött videók oldalára navigál
        ui.button("Feltöltött videók", on_click=lambda: ui.navigate.to("/uploaded_videos")).classes("mt-4")
        # Elemzés konfigurációs oldal -> elemzés konfigurációs oldalra navigál
        ui.button("Videóelemzés", on_click=lambda: ui.navigate.to("/analysis_config")).classes("mt-4")
        # Vissza a kezdőlapra gomb -> kezdőlapra navigál
        ui.button("Vissza a kezdőlapra", on_click=lambda: ui.navigate.to("/")).classes("mt-4")

# --- GUI indítása natív módban, teljes képernyőn ---
app.add_static_files('/videos', INPUT_DIR)
ui.run(native=True, fullscreen=True, title="Futballanalízis", dark=True)