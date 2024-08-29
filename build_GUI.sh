micromamba activate vegvisir
pyi-makespec Dromi_GUI.py --paths dromi/src #re-writes the .spec file, so careful
pyinstaller Dromi_GUI.spec
./dist/Dromi_GUI/Dromi_GUI