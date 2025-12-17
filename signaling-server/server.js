const WebSocket = require('ws');
const http = require('http');
const os = require('os');

const PORT = process.env.PORT || 8080;

// Get local IP addresses
function getLocalIPs() {
  const interfaces = os.networkInterfaces();
  const addresses = [];
  for (const name of Object.keys(interfaces)) {
    for (const iface of interfaces[name]) {
      if (iface.family === 'IPv4' && !iface.internal) {
        addresses.push(iface.address);
      }
    }
  }
  return addresses;
}

// Create HTTP server for health check
const server = http.createServer((req, res) => {
  if (req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok', clients: clients.size }));
  } else {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(`
      <html>
        <head><title>WebLLM Signaling Server</title></head>
        <body style="font-family: sans-serif; padding: 20px;">
          <h1>🔗 WebLLM Signaling Server</h1>
          <p>Status: <strong style="color: green;">Running</strong></p>
          <p>Connected clients: <strong>${clients.size}</strong></p>
          <p>WebSocket URL: <code>ws://${getLocalIPs()[0] || 'localhost'}:${PORT}</code></p>
        </body>
      </html>
    `);
  }
});

const wss = new WebSocket.Server({ server });

// Store connected clients: Map<deviceId, { ws, name, roomToken }>
const clients = new Map();

// Store rooms: Map<token, Set<deviceId>>
const rooms = new Map();

wss.on('connection', (ws) => {
  let clientId = null;

  ws.on('message', (data) => {
    try {
      const message = JSON.parse(data.toString());
      handleMessage(ws, message);
    } catch (e) {
      console.error('Invalid message:', e);
    }
  });

  ws.on('close', () => {
    if (clientId && clients.has(clientId)) {
      const client = clients.get(clientId);
      
      // Remove from room
      if (client.roomToken && rooms.has(client.roomToken)) {
        rooms.get(client.roomToken).delete(clientId);
        
        // Notify others in room
        broadcastToRoom(client.roomToken, {
          type: 'peer-left',
          peerId: clientId,
          peerName: client.name,
        }, clientId);
      }
      
      clients.delete(clientId);
      console.log(`Client disconnected: ${clientId}`);
    }
  });

  function handleMessage(ws, message) {
    switch (message.type) {
      case 'register':
        clientId = message.deviceId;
        clients.set(clientId, {
          ws,
          name: message.deviceName,
          roomToken: null,
        });
        console.log(`Client registered: ${clientId} (${message.deviceName})`);
        
        ws.send(JSON.stringify({ type: 'registered', deviceId: clientId }));
        break;


      case 'join-room':
        if (!clientId) return;
        
        const token = message.token;
        const client = clients.get(clientId);
        
        // Leave previous room
        if (client.roomToken && rooms.has(client.roomToken)) {
          rooms.get(client.roomToken).delete(clientId);
        }
        
        // Join new room
        if (!rooms.has(token)) {
          rooms.set(token, new Set());
        }
        rooms.get(token).add(clientId);
        client.roomToken = token;
        
        // Get other peers in room
        const peersInRoom = [];
        for (const peerId of rooms.get(token)) {
          if (peerId !== clientId) {
            const peer = clients.get(peerId);
            if (peer) {
              peersInRoom.push({ id: peerId, name: peer.name });
            }
          }
        }
        
        // Send room info to joining client
        ws.send(JSON.stringify({
          type: 'room-joined',
          token,
          peers: peersInRoom,
        }));
        
        // Notify others
        broadcastToRoom(token, {
          type: 'peer-joined',
          peerId: clientId,
          peerName: client.name,
        }, clientId);
        
        console.log(`Client ${clientId} joined room ${token}`);
        break;

      case 'leave-room':
        if (!clientId) return;
        const leavingClient = clients.get(clientId);
        
        if (leavingClient.roomToken && rooms.has(leavingClient.roomToken)) {
          const roomToken = leavingClient.roomToken;
          rooms.get(roomToken).delete(clientId);
          
          broadcastToRoom(roomToken, {
            type: 'peer-left',
            peerId: clientId,
            peerName: leavingClient.name,
          }, clientId);
          
          leavingClient.roomToken = null;
        }
        break;

      case 'signal':
        // WebRTC signaling (offer, answer, ice-candidate)
        if (!clientId) return;
        
        const targetClient = clients.get(message.targetId);
        if (targetClient && targetClient.ws.readyState === WebSocket.OPEN) {
          targetClient.ws.send(JSON.stringify({
            type: 'signal',
            signalType: message.signalType,
            data: message.data,
            fromId: clientId,
            fromName: clients.get(clientId)?.name,
          }));
        }
        break;

      case 'broadcast':
        // Broadcast message to all peers in room
        if (!clientId) return;
        const senderClient = clients.get(clientId);
        
        if (senderClient.roomToken) {
          broadcastToRoom(senderClient.roomToken, {
            type: 'broadcast',
            data: message.data,
            fromId: clientId,
            fromName: senderClient.name,
          }, clientId);
        }
        break;

      case 'direct':
        // Direct message to specific peer
        if (!clientId) return;
        
        const directTarget = clients.get(message.targetId);
        if (directTarget && directTarget.ws.readyState === WebSocket.OPEN) {
          directTarget.ws.send(JSON.stringify({
            type: 'direct',
            data: message.data,
            fromId: clientId,
            fromName: clients.get(clientId)?.name,
          }));
        }
        break;
    }
  }
});

function broadcastToRoom(token, message, excludeId = null) {
  if (!rooms.has(token)) return;
  
  for (const peerId of rooms.get(token)) {
    if (peerId !== excludeId) {
      const peer = clients.get(peerId);
      if (peer && peer.ws.readyState === WebSocket.OPEN) {
        peer.ws.send(JSON.stringify(message));
      }
    }
  }
}

server.listen(PORT, '0.0.0.0', () => {
  const ips = getLocalIPs();
  console.log('\n🔗 WebLLM Signaling Server Started\n');
  console.log(`   Local:    http://localhost:${PORT}`);
  ips.forEach(ip => {
    console.log(`   Network:  http://${ip}:${PORT}`);
    console.log(`   WS URL:   ws://${ip}:${PORT}`);
  });
  console.log('\n   Share the WS URL with other devices on your network.\n');
});
