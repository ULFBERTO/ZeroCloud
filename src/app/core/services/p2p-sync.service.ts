import { Injectable, signal, computed } from '@angular/core';
import Peer, { DataConnection } from 'peerjs';

export interface PeerDevice {
  id: string;
  name: string;
  status: 'connecting' | 'connected' | 'error';
  connection?: DataConnection;
}

export type ConnectionMode = 'disconnected' | 'connecting' | 'hosting' | 'connected';

const DEVICE_NAME_KEY = 'webllm_device_name';

@Injectable({ providedIn: 'root' })
export class P2PSyncService {
  private peer: Peer | null = null;
  private connections = new Map<string, DataConnection>();

  private deviceName: string;

  private readonly _peerId = signal<string | null>(null);
  private readonly _roomCode = signal<string | null>(null);
  private readonly _connectionMode = signal<ConnectionMode>('disconnected');
  private readonly _peers = signal<Map<string, PeerDevice>>(new Map());
  private readonly _error = signal<string | null>(null);

  readonly peerId = this._peerId.asReadonly();
  readonly roomCode = this._roomCode.asReadonly();
  readonly connectionMode = this._connectionMode.asReadonly();
  readonly peers = computed(() => Array.from(this._peers().values()));
  readonly connectedPeers = computed(() => 
    this.peers().filter(p => p.status === 'connected')
  );
  readonly isConnected = computed(() => this.connectedPeers().length > 0);
  readonly error = this._error.asReadonly();

  constructor() {
    this.deviceName = this.getDeviceName();
  }

  private getDeviceName(): string {
    let name = localStorage.getItem(DEVICE_NAME_KEY);
    if (!name) {
      name = `User-${Math.random().toString(36).substring(2, 6).toUpperCase()}`;
      localStorage.setItem(DEVICE_NAME_KEY, name);
    }
    return name;
  }

  setDeviceName(name: string): void {
    this.deviceName = name;
    localStorage.setItem(DEVICE_NAME_KEY, name);
  }

  getDeviceInfo(): { id: string | null; name: string } {
    return { id: this._peerId(), name: this.deviceName };
  }


  // === Crear sala (Host) ===

  async createRoom(): Promise<string> {
    this._connectionMode.set('connecting');
    this._error.set(null);

    return new Promise((resolve, reject) => {
      // Generar código de sala corto
      const roomCode = this.generateRoomCode();
      const peerId = `webllm-${roomCode}`;

      this.peer = new Peer(peerId, {
        debug: 0,
      });

      this.peer.on('open', (id) => {
        this._peerId.set(id);
        this._roomCode.set(roomCode);
        this._connectionMode.set('hosting');
        resolve(roomCode);
      });

      this.peer.on('connection', (conn) => {
        this.handleIncomingConnection(conn);
      });

      this.peer.on('error', (err) => {
        console.error('PeerJS error:', err);
        if (err.type === 'unavailable-id') {
          // El código ya existe, generar otro
          this.peer?.destroy();
          this.createRoom().then(resolve).catch(reject);
        } else {
          this._error.set(`Error: ${err.message}`);
          this._connectionMode.set('disconnected');
          reject(err);
        }
      });

      this.peer.on('disconnected', () => {
        this._connectionMode.set('disconnected');
      });
    });
  }

  // === Unirse a sala ===

  async joinRoom(roomCode: string): Promise<void> {
    this._connectionMode.set('connecting');
    this._error.set(null);

    return new Promise((resolve, reject) => {
      const myPeerId = `webllm-${roomCode}-${Date.now()}`;
      const hostPeerId = `webllm-${roomCode}`;

      this.peer = new Peer(myPeerId, {
        debug: 0,
      });

      this.peer.on('open', (id) => {
        this._peerId.set(id);
        this._roomCode.set(roomCode);

        // Conectar al host
        const conn = this.peer!.connect(hostPeerId, {
          metadata: { name: this.deviceName },
          reliable: true,
        });

        conn.on('open', () => {
          this.handleOutgoingConnection(conn, hostPeerId);
          this._connectionMode.set('connected');
          resolve();
        });

        conn.on('error', (err) => {
          this._error.set('No se pudo conectar a la sala');
          this._connectionMode.set('disconnected');
          reject(err);
        });
      });

      this.peer.on('connection', (conn) => {
        this.handleIncomingConnection(conn);
      });

      this.peer.on('error', (err) => {
        console.error('PeerJS error:', err);
        if (err.type === 'peer-unavailable') {
          this._error.set('Sala no encontrada. Verifica el código.');
        } else {
          this._error.set(`Error: ${err.message}`);
        }
        this._connectionMode.set('disconnected');
        reject(err);
      });
    });
  }

  private generateRoomCode(): string {
    return Math.random().toString(36).substring(2, 8).toUpperCase();
  }


  // === Manejo de conexiones ===

  private handleIncomingConnection(conn: DataConnection): void {
    const peerId = conn.peer;
    const peerName = (conn.metadata?.name as string) || 'Unknown';

    this._peers.update((peers) => {
      const newPeers = new Map(peers);
      newPeers.set(peerId, {
        id: peerId,
        name: peerName,
        status: 'connecting',
        connection: conn,
      });
      return newPeers;
    });

    conn.on('open', () => {
      this.connections.set(peerId, conn);
      this._peers.update((peers) => {
        const newPeers = new Map(peers);
        const peer = newPeers.get(peerId);
        if (peer) {
          newPeers.set(peerId, { ...peer, status: 'connected' });
        }
        return newPeers;
      });

      // Enviar info propia
      conn.send({ type: 'hello', name: this.deviceName });
    });

    conn.on('data', (data) => {
      this.handleMessage(data as Record<string, unknown>, peerId);
    });

    conn.on('close', () => {
      this.removePeer(peerId);
    });

    conn.on('error', () => {
      this._peers.update((peers) => {
        const newPeers = new Map(peers);
        const peer = newPeers.get(peerId);
        if (peer) {
          newPeers.set(peerId, { ...peer, status: 'error' });
        }
        return newPeers;
      });
    });
  }

  private handleOutgoingConnection(conn: DataConnection, peerId: string): void {
    this.connections.set(peerId, conn);

    this._peers.update((peers) => {
      const newPeers = new Map(peers);
      newPeers.set(peerId, {
        id: peerId,
        name: 'Host',
        status: 'connected',
        connection: conn,
      });
      return newPeers;
    });

    conn.on('data', (data) => {
      this.handleMessage(data as Record<string, unknown>, peerId);
    });

    conn.on('close', () => {
      this.removePeer(peerId);
    });

    // Enviar info propia
    conn.send({ type: 'hello', name: this.deviceName });
  }

  private removePeer(peerId: string): void {
    this.connections.delete(peerId);
    this._peers.update((peers) => {
      const newPeers = new Map(peers);
      newPeers.delete(peerId);
      return newPeers;
    });
  }


  // === Mensajes ===

  private handleMessage(message: Record<string, unknown>, fromId: string): void {
    switch (message['type']) {
      case 'hello':
        this._peers.update((peers) => {
          const newPeers = new Map(peers);
          const peer = newPeers.get(fromId);
          if (peer) {
            newPeers.set(fromId, { ...peer, name: message['name'] as string });
          }
          return newPeers;
        });
        break;

      case 'inference-task':
        window.dispatchEvent(
          new CustomEvent('p2p-inference-task', {
            detail: {
              taskId: message['taskId'],
              prompt: message['prompt'],
              fromId,
            },
          })
        );
        break;

      case 'inference-result':
        window.dispatchEvent(
          new CustomEvent('p2p-inference-result', {
            detail: {
              taskId: message['taskId'],
              result: message['result'],
              fromId,
              fromName: this._peers().get(fromId)?.name || 'Unknown',
            },
          })
        );
        break;

      case 'inference-stream':
        window.dispatchEvent(
          new CustomEvent('p2p-inference-stream', {
            detail: {
              taskId: message['taskId'],
              chunk: message['chunk'],
              fromId,
            },
          })
        );
        break;
    }
  }

  sendToPeer(peerId: string, message: object): boolean {
    const conn = this.connections.get(peerId);
    if (conn?.open) {
      conn.send(message);
      return true;
    }
    return false;
  }

  broadcastToPeers(message: object): void {
    this.connections.forEach((conn) => {
      if (conn.open) {
        conn.send(message);
      }
    });
  }

  sendInferenceTask(prompt: string): string {
    const taskId = crypto.randomUUID();
    this.broadcastToPeers({
      type: 'inference-task',
      taskId,
      prompt,
    });
    return taskId;
  }

  sendInferenceResult(taskId: string, targetId: string, result: string): void {
    this.sendToPeer(targetId, {
      type: 'inference-result',
      taskId,
      result,
    });
  }

  // === Desconexión ===

  disconnect(): void {
    this.connections.forEach((conn) => conn.close());
    this.connections.clear();
    this.peer?.destroy();
    this.peer = null;
    this._peerId.set(null);
    this._roomCode.set(null);
    this._connectionMode.set('disconnected');
    this._peers.set(new Map());
    this._error.set(null);
  }
}
