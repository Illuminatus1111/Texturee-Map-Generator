const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  selectImage: () => ipcRenderer.invoke('select-image'),
  processImage: (data) => ipcRenderer.invoke('process-image', data),
  saveOutput: (outputPath) => ipcRenderer.invoke('save-output', outputPath)
});
