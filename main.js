const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const url = require('url');
const fs = require('fs');
const { spawn } = require('child_process');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  const startUrl = url.format({
    pathname: path.join(__dirname, 'build', 'index.html'),
    protocol: 'file:',
    slashes: true,
  });
  mainWindow.loadURL(startUrl);  
  
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
app.on('activate', () => {
  if (mainWindow === null) createWindow();
});

// Image selection
ipcMain.handle('select-image', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [{ name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'bmp', 'tga', 'tiff'] }],
  });
  return result.canceled ? null : result.filePaths[0];
});

// Image processing
ipcMain.handle('process-image', async (event, { imagePath, selectedMaps, settings }) => {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(__dirname, 'backend/process_image.py');
    const pythonPath = 'python'; // change to full path if needed

    const proc = spawn(pythonPath, [
      scriptPath,
      imagePath,
      JSON.stringify(selectedMaps),
      JSON.stringify(settings),
    ]);

    let output = '';
    let error = '';

    proc.stdout.on('data', (data) => (output += data));
    proc.stderr.on('data', (data) => (error += data));

    proc.on('close', (code) => {
      if (code === 0) {
        try {
          resolve(JSON.parse(output));
        } catch (err) {
          reject(`Invalid JSON from Python: ${err.message}`);
        }
      } else {
        reject(`Python error: ${error}`);
      }
    });
  });
});
