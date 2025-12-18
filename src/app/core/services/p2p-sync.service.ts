import { Injectable, signal, computed } from '@angular/core';
import Peer, { DataConnection } from 'peerjs';

export interface PeerDevice {
  id: string;
  name: string;
  status: 'connecting' | 'connected' | 'error';
  connection?: DataConnection;
  // Estadísticas de carga
  tasksCompleted: number;
  tasksInProgress: number;
  lastTaskTime?: number; // ms que tardó la última tarea
}

export interface TaskStats {
  taskId: string;
  prompt: string;
  assignedTo: string;
  startTime: number;
  endTime?: number;
  status: 'pending' | 'processing' | 'completed';
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
  private readonly _tasks = signal<Map<string, TaskStats>>(new Map());
  private readonly _myStats = signal<{ tasksCompleted: number; tasksInProgress: number; totalTime: number }>({
    tasksCompleted: 0,
    tasksInProgress: 0,
    totalTime: 0,
  });

  readonly peerId = this._peerId.asReadonly();
  readonly roomCode = this._roomCode.asReadonly();
  readonly connectionMode = this._connectionMode.asReadonly();
  readonly peers = computed(() => Array.from(this._peers().values()));
  readonly connectedPeers = computed(() => 
    this.peers().filter(p => p.status === 'connected')
  );
  readonly isConnected = computed(() => this.connectedPeers().length > 0);
  readonly error = this._error.asReadonly();
  readonly tasks = computed(() => Array.from(this._tasks().values()));
  readonly myStats = this._myStats.asReadonly();
  
  // Estadísticas agregadas
  readonly totalTasksCompleted = computed(() => 
    this.peers().reduce((sum, p) => sum + p.tasksCompleted, 0) + this._myStats().tasksCompleted
  );
  readonly totalTasksInProgress = computed(() =>
    this.peers().reduce((sum, p) => sum + p.tasksInProgress, 0) + this._myStats().tasksInProgress
  );

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
        tasksCompleted: 0,
        tasksInProgress: 0,
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
        tasksCompleted: 0,
        tasksInProgress: 0,
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
        // Incrementar tareas en progreso (soy worker)
        this._myStats.update(s => ({ ...s, tasksInProgress: s.tasksInProgress + 1 }));
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

      case 'inference-result': {
        const taskId = message['taskId'] as string;
        // Actualizar estadísticas del peer que completó
        this._peers.update((peers) => {
          const newPeers = new Map(peers);
          const peer = newPeers.get(fromId);
          if (peer) {
            newPeers.set(fromId, {
              ...peer,
              tasksCompleted: peer.tasksCompleted + 1,
              tasksInProgress: Math.max(0, peer.tasksInProgress - 1),
            });
          }
          return newPeers;
        });
        // Actualizar tarea
        this._tasks.update((tasks) => {
          const newTasks = new Map(tasks);
          const task = newTasks.get(taskId);
          if (task) {
            newTasks.set(taskId, { ...task, status: 'completed', endTime: Date.now() });
          }
          return newTasks;
        });
        window.dispatchEvent(
          new CustomEvent('p2p-inference-result', {
            detail: {
              taskId,
              result: message['result'],
              fromId,
              fromName: this._peers().get(fromId)?.name || 'Unknown',
            },
          })
        );
        break;
      }

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

      case 'task-started':
        // El peer empezó a procesar
        this._peers.update((peers) => {
          const newPeers = new Map(peers);
          const peer = newPeers.get(fromId);
          if (peer) {
            newPeers.set(fromId, { ...peer, tasksInProgress: peer.tasksInProgress + 1 });
          }
          return newPeers;
        });
        break;

      case 'stats-update':
        // Actualizar estadísticas del peer
        this._peers.update((peers) => {
          const newPeers = new Map(peers);
          const peer = newPeers.get(fromId);
          if (peer) {
            newPeers.set(fromId, {
              ...peer,
              tasksCompleted: (message['completed'] as number) || peer.tasksCompleted,
              tasksInProgress: (message['inProgress'] as number) || 0,
              lastTaskTime: message['lastTime'] as number | undefined,
            });
          }
          return newPeers;
        });
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
    
    // Registrar tarea
    this._tasks.update((tasks) => {
      const newTasks = new Map(tasks);
      newTasks.set(taskId, {
        taskId,
        prompt: prompt.substring(0, 50) + (prompt.length > 50 ? '...' : ''),
        assignedTo: 'broadcast',
        startTime: Date.now(),
        status: 'pending',
      });
      return newTasks;
    });

    this.broadcastToPeers({
      type: 'inference-task',
      taskId,
      prompt,
    });
    return taskId;
  }

  sendInferenceResult(taskId: string, targetId: string, result: string): void {
    // Actualizar mis estadísticas (completé una tarea)
    this._myStats.update(s => ({
      ...s,
      tasksCompleted: s.tasksCompleted + 1,
      tasksInProgress: Math.max(0, s.tasksInProgress - 1),
    }));

    // Notificar al host
    this.sendToPeer(targetId, {
      type: 'inference-result',
      taskId,
      result,
    });

    // Enviar actualización de stats
    this.broadcastToPeers({
      type: 'stats-update',
      completed: this._myStats().tasksCompleted,
      inProgress: this._myStats().tasksInProgress,
    });
  }

  // Notificar que empecé a procesar
  notifyTaskStarted(taskId: string): void {
    this.broadcastToPeers({ type: 'task-started', taskId });
  }

  // Obtener estadísticas de carga por peer
  getPeerLoadStats(): { peerId: string; name: string; load: number; completed: number }[] {
    return this.connectedPeers().map(p => ({
      peerId: p.id,
      name: p.name,
      load: p.tasksInProgress,
      completed: p.tasksCompleted,
    }));
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
    this._tasks.set(new Map());
    this._myStats.set({ tasksCompleted: 0, tasksInProgress: 0, totalTime: 0 });
  }
}
