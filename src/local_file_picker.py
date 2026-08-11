import platform
from pathlib import Path
from paths import *

from nicegui import events, ui


class local_file_picker(ui.dialog):

    def __init__(self, directory: str, *,
                 upper_limit: str | None = ...,) -> None:
        """Local File Picker

        This is a simple file picker that allows you to select a file from the local filesystem where NiceGUI is running.

        :param directory: The directory to start in.
        :param upper_limit: The directory to stop at (None: no limit, default: same as the starting directory).
        :param multiple: Whether to allow multiple files to be selected.
        :param show_hidden_files: Whether to show hidden files.
        """
        super().__init__()

        self.directory = directory
        with self, ui.card().classes('q-pa-md q-ma-sm bg-primary text-white'):
            ui.label(f"Select a file from {self.directory}")
            self.grid = ui.aggrid({
                'columnDefs': [
                    {'headerName': 'File', 'field': 'file'}
                ],
                'rowData': [
                    {'file': f.name} for f in Path(self.directory).iterdir()
                    ],
                'rowSelection': {'mode': 'multiRow'}
            })
            with ui.row():
                self.cancel_button = ui.button("Cancel", on_click=self.close)
                self.select_button = ui.button("Select", on_click=self._select)
        self.update_grid()

    def update_grid(self):
        self.grid.update()

    async def _select(self):
        selected_rows = await self.grid.get_selected_rows()
        self.submit([str(CONFIG_FOLDER / row['file']) for row in selected_rows])