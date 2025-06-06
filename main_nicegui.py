from nicegui import ui, events, app
from pathlib import Path
from main_restructured import run_analysis_pipeline
import asyncio
from concurrent.futures import ThreadPoolExecutor
import shutil
import os

# ----------------
# --- VÁLTOZÓK ---
# ----------------

# Feltöltött videók könyvtára
INPUT_DIR = Path("input_videos")
INPUT_DIR.mkdir(exist_ok=True)

# Elemzett videók könyvtára
OUTPUT_DIR = Path("output_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

# Kiválasztott videó tárolása
selected_video = None
# Kiválasztott elemzett videó tárolása
selected_analyzed_video = None
# Log terület a visszajelzésekhez
ui_log_area = None

# Aszinkron futtatáshoz szükséges executor
executor = ThreadPoolExecutor()

# ----------------
# --- FUNKCIÓK ---
# ----------------

# Aszinkron függvény a videóelemzés futtatására háttérben
async def run_in_thread(func):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func)

# Aszinkron függvény a videóelemzés futtatására visszajelzéssel
async def run_pipeline_with_feedback(video_path: str, loader_dialog, container_to_hide):
    # Töltőképernyő mögötti konténer elrejtése
    container_to_hide.set_visibility(False)
    # Log terület törlése az elemzés kezdetekor
    if ui_log_area:
        ui_log_area.value = ""
    # Töltőképernyő megnyitása
    loader_dialog.open()
    # Elemzés futtatása a háttérben
    await run_in_thread(lambda: run_analysis_pipeline(video_path))
    # Töltőképernyő bezárása
    loader_dialog.close()
    ui.notify("Elemzés befejeződött!")
    ui.navigate.to('/analyzed_videos')

# Log üzenetek megjelenítése a GUI-ban
def gui_log(msg: str):
    global ui_log_area
    print(msg)  # konzolra továbbra is megy
    if ui_log_area:
        def update():
            ui_log_area.value += f"{msg}\n"
            ui_log_area.run_method('scrollTop', ui_log_area.element.scrollHeight)
        ui.run_later(update)  # 👈 GUI frissítés fő szálon

def open_video_file(path):
    try:
        os.startfile(Path(path).resolve())
    except Exception as e:
        ui.notify(f"Hiba a videó megnyitásakor: {e}", color="red")

# Videó feltöltése és mentése a megadott könyvtárba
def save_uploaded_video(e: events.UploadEventArguments):
    target_path = INPUT_DIR / e.name
    # Videó átmásolása a megadott könyvtárba
    with open(target_path, 'wb') as f:
        f.write(e.content.read())
    ui.notify(f"Videó feltöltve: {e.name}")

# Adott videófájl törlése gombnyomásra (feltöltött videók)
def delete_video_file(file_path: Path):
    if file_path.exists():
        file_path.unlink()
        ui.notify(f"Törölve: {file_path.name}")
        # Oldal frissítése a törlés után
        ui.navigate.to('/uploaded_videos')

# Adott elemzett videófájl részleteinek megjelenítése gombnyomásra
def show_analyzed_video_details(video_file: Path):
    global selected_analyzed_video
    selected_analyzed_video = video_file
    ui.navigate.to('/analyzed_video_detail')

# Adott elemzett videófájlhoz tartozó mappa törlése gombnyomásra
def delete_analyzed_video_file(video_path: Path):
    try:
        # Az elemzett videó mappájának törlése
        folder_path = video_path.parent
        shutil.rmtree(folder_path)
        ui.notify(f"Törölve: {folder_path.name}")
        ui.navigate.to('/analyzed_videos')
    except Exception as e:
        ui.notify(f"A fájlt nem lehet törölni, mert még használatban van!", color='orange')

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
        ui.button("Indítás", on_click=lambda: ui.navigate.to("/main_page")).classes("w-48 text-lg")
        # Kilépés gomb -> alkalmazás leállítása
        ui.button("Kilépés", on_click=app.shutdown).classes("w-48 text-lg")

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

# --- Elemzett videók oldala ("/analyzed_videos") ---
@ui.page("/analyzed_videos")
def analyzed_videos_page():
    with ui.column().classes("absolute-center items-center gap-4"):
        ui.label("Elemzett videók").classes("text-2xl font-semibold")
        video_dirs = [p for p in OUTPUT_DIR.iterdir() if p.is_dir() and p.name.startswith("annotated_")]

        if not video_dirs:
            ui.label("Az elemzett videók mappája üres!").classes("text-white")
        else:
            with ui.row().classes("justify-center flex-wrap gap-8"):
                for video_dir in video_dirs:
                    video_name = video_dir.name.replace("annotated_", "")
                    video_path = video_dir / f"annotated_{video_name}.mp4"
                    with ui.column().classes("items-center cursor-pointer").on("click", lambda f=video_path: show_analyzed_video_details(f)):
                        ui.icon("folder").classes("text-yellow-400 text-6xl")
                        ui.label(video_name + ".mp4").classes("text-sm text-center text-white mt-2 max-w-48 truncate")

        ui.button("Vissza", on_click=lambda: ui.navigate.to("/main_page")).classes("mt-4")

@ui.page("/analyzed_video_detail")
def analyzed_video_detail_page():
 
    with ui.column().classes("absolute-center items-center gap-4"):
        # Statikus előnézet
        ui.label("Elemzett videó részletei").classes("text-2xl font-semibold")

        # Elemzett videó thumbnailjének megjelenítése
        thumbnail_path = f"/analyzed_videos/{selected_analyzed_video.parent.name}/thumbnail/thumbnail.jpg"
        ui.image(thumbnail_path).classes("w-96 h-56 rounded shadow object-cover")

        # Elemzett videó neve
        ui.label(selected_analyzed_video.name).classes("text-white text-center")

        # Gombok
        with ui.column().classes("gap-2 w-64"):
            ui.button("Videó megnyitása", on_click=lambda: open_video_file(selected_analyzed_video)).classes("w-full")
            # ui.button("Statisztikák").classes("w-full")
            # ui.button("Hőtérkép").classes("w-full")
            # ui.button("Grafikonok").classes("w-full")
            ui.button("Törlés", on_click=lambda: delete_analyzed_video_file(selected_analyzed_video)).props('color="red" icon="delete"').classes("w-full")

        ui.button("Vissza az elemzett videókhoz", on_click=lambda: ui.navigate.to("/analyzed_videos")).classes("mt-6")

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
            # Videó feltöltése gomb
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
    global ui_log_area
    
    page_container = ui.column().classes("absolute-center items-center gap-4")

    with page_container:
        # Elemzés konfigurációs oldal címe
        ui.label("Videó elemzése").classes("text-2xl font-semibold")
        
        # Kiválasztott videó megjelenítése
        if selected_video:
            ui.markdown(f"**Kiválasztott videó:** {selected_video.name}").classes("text-center text-green-400")
        else:
            ui.markdown("Még nincs videó kiválasztva az elemzéshez.").classes("text-center text-orange-400")
        
        # Videó kiválasztása gomb
        ui.button("Videó kiválasztása", 
                 on_click=lambda: ui.navigate.to("/select_video_for_analysis")).classes("w-48 text-lg")
        
        # Töltőképernyő dialógus létrehozása
        loader_dialog = ui.dialog().classes("bg-transparent")
        with loader_dialog:
            with ui.column().classes("items-center gap-4"):
                ui.spinner(size="lg", color="primary")
                ui.label("Elemzés folyamatban...").classes("text-white text-lg text-center")
                
                # Folyamatnapló terület a töltőképernyőn
                ui_log_area = ui.textarea(label="Folyamatnapló") \
                    .classes("w-96 h-48 bg-black text-white text-xs p-2") \
                    .props("filled readonly dense")
        
        # Elemzés indítása gomb (csak ha van kiválasztott videó)
        if selected_video:
            ui.button("Elemzés indítása", 
                     on_click=lambda: run_pipeline_with_feedback(str(selected_video), loader_dialog, page_container)
                     ).classes("w-48 text-lg mt-2")
            
        # Vissza a menübe gomb -> visszalépés a főoldalra
        ui.button("Vissza", on_click=lambda: ui.navigate.to("/main_page")).classes("mt-4")

# --- Elemezni kívánt videó kiválasztása oldal ---
@ui.page("/select_video")
def select_video_page():
    with ui.column().classes("absolute-center items-center gap-4"):
        # Videó kiválasztása oldal címe
        ui.label("Videó kiválasztása").classes("text-2xl font-semibold")
        # Videó kiválasztása gomb --> videó kiválasztása oldalra navigál
        ui.button("Videó kiválasztása", on_click=lambda: ui.navigate.to("/analysis_config")).classes("w-48 text-lg")
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
        # Elemzett videók -> elemzett videók oldalára navigál
        ui.button("Elemzett videók", on_click=lambda: ui.navigate.to("/analyzed_videos")).classes("mt-4")
        # Vissza a kezdőlapra gomb -> kezdőlapra navigál
        ui.button("Vissza a kezdőlapra", on_click=lambda: ui.navigate.to("/")).classes("mt-4")

# --- GUI indítása natív módban, teljes képernyőn ---
app.add_static_files('/videos', INPUT_DIR)
app.add_static_files('/analyzed_videos', OUTPUT_DIR)
ui.run(native=True, fullscreen=True, title="Futballanalízis", dark=True)