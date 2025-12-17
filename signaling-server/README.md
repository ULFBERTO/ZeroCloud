# WebLLM Signaling Server

Servidor de señalización para la sincronización P2P de WebLLM Chat.

## Instalación

```bash
cd signaling-server
npm install
```

## Uso

```bash
npm start
```

El servidor mostrará las URLs disponibles:

```
🔗 WebLLM Signaling Server Started

   Local:    http://localhost:8080
   Network:  http://192.168.1.X:8080
   WS URL:   ws://192.168.1.X:8080

   Share the WS URL with other devices on your network.
```

## Configuración en la App

1. Abre el chat y haz click en el botón "🔗 P2P"
2. En "Servidor de señalización", ingresa la URL WebSocket (ej: `ws://192.168.1.X:8080`)
3. Click en "Conectar"
4. Crea una sala o únete a una existente con el código

## Cómo funciona

1. **Señalización (WebSocket):** El servidor facilita el intercambio inicial de información entre peers
2. **Conexión P2P (WebRTC):** Una vez conectados, los datos fluyen directamente entre navegadores
3. **Salas:** Los dispositivos se agrupan por código de sala para la inferencia distribuida

## Arquitectura

```
┌─────────────┐     WebSocket      ┌─────────────────┐     WebSocket      ┌─────────────┐
│  Browser A  │◄──────────────────►│ Signaling Server│◄──────────────────►│  Browser B  │
└─────────────┘                    └─────────────────┘                    └─────────────┘
       │                                                                         │
       │                         WebRTC DataChannel                              │
       └─────────────────────────────────────────────────────────────────────────┘
                                  (P2P directo)
```

## Puerto

Por defecto usa el puerto 8080. Puedes cambiarlo con la variable de entorno:

```bash
PORT=3000 npm start
```
