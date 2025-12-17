import { Injectable, signal, computed } from '@angular/core';

export interface PeerDevice {
  id: string;
  name: string;
  status: 'discovered' | 'pending' | 'connecting' | 'connected' | 'error';
  lastSeen: Date;
  isHost: boolean;
}

export interface SyncRequest {
  fromId: string;
  fromName: string;
  token: string;
  timestamp: number;
}

export type ConnectionMode = 'disconnected' | 'connecting' | 'connected';

const DEVICE_ID_KEY = 'webllm_device_id';
const DEVICE_NAME_KEY = 'webllm_device_name';
const SERVER_URL_KEY = 'webllm_signaling_server';
const DEFAULT_SERVER = 'ws://localhost:8080';

@Injectable({ providedIn: 'root' })
export class P2PSyncService {
  private ws: WebSocket | null = null;
  private peerConnections = new Map<string, RTCPeerConnection>();
  private dataChannels = new Map<string, RTCDataChannel>();

  private readonly deviceId: string;
  private deviceName: string;

  private readonly _serverUrl = signal<string>(
    localStorage.getItem(SERVER_URL_KEY) || DEFAULT_SERVER
  );
  private readonly _connectionMode = signal<ConnectionMode>('disconnected');
  private readonly _peers = signal<Map<string, PeerDevice>>(new Map());
  private readonly _connectedPeers = signal<Set<string>>(new Set());
  private readonly _syncToken = signal<string | null>(null);
  private readonly _error = signal<string | null>(null);

  readonly serverUrl = this._serverUrl.asReadonly();
  readonly connectionMode = this._connectionMode.asReadonly();
  readonly peers = computed(() => Array.from(this._peers().values()));
  readonly connectedPeers = computed(() => Array.from(this._connectedPeers()));
  readonly syncToken = this._syncToken.asReadonly();
  readonly isConnected = computed(() => this._connectedPeers().size > 0);
  readonly error = this._error.asReadonly();

  constructor() {
    this.deviceId = this.getOrCreateDeviceId();
    this.deviceName = this.getDeviceName();
  }

  private getOrCreateDeviceId(): string {
    let id = localStorage.getItem(DEVICE_ID_KEY);
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem(DEVICE_ID_KEY, id);
    }
    return id;
  }

  private getDeviceName(): string {
    let name = localStorage.getItem(DEVICE_NAME_KEY);
    if (!name) {
      const platform = navigator.platform || 'Device';
      name = `${platform}-${this.deviceId.slice(0, 4)}`;
      localStorage.setItem(DEVICE_NAME_KEY, name);
    }
    return name;
  }

  setDeviceName(name: string): void {
    this.deviceName = name;
    localStorage.setItem(DEVICE_NAME_KEY, name);
  }

  setServerUrl(url: string): void {
    this._serverUrl.set(url);
    localStorage.setItem(SERVER_URL_KEY, url);
  }

  getDeviceInfo(): { id: string; name: string } {
    return { id: this.deviceId, name: this.deviceName };
  }


  // === WebSocket Connection ===

  connectToServer(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      this._connectionMode.set('connecting');
      this._error.set(null);

      try {
        this.ws = new WebSocket(this._serverUrl());

        this.ws.onopen = () => {
          this.ws!.send(JSON.stringify({
            type: 'register',
            deviceId: this.deviceId,
            deviceName: this.deviceName,
          }));
        };

        this.ws.onmessage = (event) => {
          const message = JSON.parse(event.data);
          this.handleServerMessage(message);

          if (message.type === 'registered') {
            this._connectionMode.set('connected');
            resolve();
          }
        };

        this.ws.onerror = () => {
          this._error.set('Error de conexión al servidor');
          this._connectionMode.set('disconnected');
          reject(new Error('WebSocket error'));
        };

        this.ws.onclose = () => {
          this._connectionMode.set('disconnected');
          this.cleanupConnections();
        };
      } catch (e) {
        this._error.set('No se pudo conectar al servidor');
        this._connectionMode.set('disconnected');
        reject(e);
      }
    });
  }

  disconnectFromServer(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.cleanupConnections();
    this._connectionMode.set('disconnected');
    this._peers.set(new Map());
    this._connectedPeers.set(new Set());
    this._syncToken.set(null);
  }

  private cleanupConnections(): void {
    this.dataChannels.forEach((dc) => dc.close());
    this.dataChannels.clear();
    this.peerConnections.forEach((pc) => pc.close());
    this.peerConnections.clear();
  }

  private sendToServer(message: object): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  private handleServerMessage(message: Record<string, unknown>): void {
    switch (message['type']) {
      case 'room-joined':
        this.handleRoomJoined(message as unknown as { peers: { id: string; name: string }[] });
        break;
      case 'peer-joined':
        this.handlePeerJoined(message as unknown as { peerId: string; peerName: string });
        break;
      case 'peer-left':
        this.handlePeerLeft(message as unknown as { peerId: string });
        break;
      case 'signal':
        this.handleSignal(message as unknown as {
          signalType: string;
          data: unknown;
          fromId: string;
          fromName: string;
        });
        break;
      case 'direct':
        this.handleDirectMessage(message as unknown as { data: unknown; fromId: string; fromName: string });
        break;
    }
  }


  // === Room Management ===

  generateSyncToken(): string {
    const token = Math.random().toString(36).substring(2, 8).toUpperCase();
    this._syncToken.set(token);
    return token;
  }

  async joinRoom(token: string): Promise<void> {
    if (this._connectionMode() !== 'connected') {
      await this.connectToServer();
    }

    this._syncToken.set(token);
    this.sendToServer({ type: 'join-room', token });
  }

  leaveRoom(): void {
    this.sendToServer({ type: 'leave-room' });
    this._syncToken.set(null);
    this._peers.set(new Map());
    this._connectedPeers.set(new Set());
    this.cleanupConnections();
  }

  private handleRoomJoined(message: { peers: { id: string; name: string }[] }): void {
    const newPeers = new Map<string, PeerDevice>();

    for (const peer of message.peers) {
      newPeers.set(peer.id, {
        id: peer.id,
        name: peer.name,
        status: 'discovered',
        lastSeen: new Date(),
        isHost: false,
      });

      // Initiate WebRTC connection to existing peers
      this.createPeerConnection(peer.id, true);
    }

    this._peers.set(newPeers);
  }

  private handlePeerJoined(message: { peerId: string; peerName: string }): void {
    this._peers.update((peers) => {
      const newPeers = new Map(peers);
      newPeers.set(message.peerId, {
        id: message.peerId,
        name: message.peerName,
        status: 'discovered',
        lastSeen: new Date(),
        isHost: false,
      });
      return newPeers;
    });
  }

  private handlePeerLeft(message: { peerId: string }): void {
    this._peers.update((peers) => {
      const newPeers = new Map(peers);
      newPeers.delete(message.peerId);
      return newPeers;
    });

    this._connectedPeers.update((peers) => {
      const newPeers = new Set(peers);
      newPeers.delete(message.peerId);
      return newPeers;
    });

    // Cleanup WebRTC
    this.dataChannels.get(message.peerId)?.close();
    this.dataChannels.delete(message.peerId);
    this.peerConnections.get(message.peerId)?.close();
    this.peerConnections.delete(message.peerId);
  }


  // === WebRTC ===

  private createPeerConnection(peerId: string, initiator: boolean): RTCPeerConnection {
    const config: RTCConfiguration = {
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' },
      ],
    };

    const pc = new RTCPeerConnection(config);
    this.peerConnections.set(peerId, pc);

    this._peers.update((peers) => {
      const newPeers = new Map(peers);
      const peer = newPeers.get(peerId);
      if (peer) {
        newPeers.set(peerId, { ...peer, status: 'connecting' });
      }
      return newPeers;
    });

    pc.onicecandidate = (event) => {
      if (event.candidate) {
        this.sendToServer({
          type: 'signal',
          signalType: 'ice-candidate',
          targetId: peerId,
          data: event.candidate,
        });
      }
    };

    pc.onconnectionstatechange = () => {
      if (pc.connectionState === 'connected') {
        this._peers.update((peers) => {
          const newPeers = new Map(peers);
          const peer = newPeers.get(peerId);
          if (peer) {
            newPeers.set(peerId, { ...peer, status: 'connected' });
          }
          return newPeers;
        });
        this._connectedPeers.update((peers) => new Set([...peers, peerId]));
      } else if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
        this._peers.update((peers) => {
          const newPeers = new Map(peers);
          const peer = newPeers.get(peerId);
          if (peer) {
            newPeers.set(peerId, { ...peer, status: 'error' });
          }
          return newPeers;
        });
      }
    };

    if (initiator) {
      const dc = pc.createDataChannel('inference');
      this.setupDataChannel(dc, peerId);

      pc.createOffer()
        .then((offer) => pc.setLocalDescription(offer))
        .then(() => {
          this.sendToServer({
            type: 'signal',
            signalType: 'offer',
            targetId: peerId,
            data: pc.localDescription,
          });
        });
    } else {
      pc.ondatachannel = (event) => {
        this.setupDataChannel(event.channel, peerId);
      };
    }

    return pc;
  }

  private setupDataChannel(dc: RTCDataChannel, peerId: string): void {
    this.dataChannels.set(peerId, dc);

    dc.onopen = () => {
      console.log(`DataChannel open with ${peerId}`);
    };

    dc.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        this.handleP2PMessage(message, peerId);
      } catch (e) {
        console.error('Invalid P2P message:', e);
      }
    };

    dc.onclose = () => {
      console.log(`DataChannel closed with ${peerId}`);
    };
  }

  private async handleSignal(message: {
    signalType: string;
    data: unknown;
    fromId: string;
    fromName: string;
  }): Promise<void> {
    let pc = this.peerConnections.get(message.fromId);

    if (message.signalType === 'offer') {
      if (!pc) {
        pc = this.createPeerConnection(message.fromId, false);
      }
      await pc.setRemoteDescription(message.data as RTCSessionDescriptionInit);
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      this.sendToServer({
        type: 'signal',
        signalType: 'answer',
        targetId: message.fromId,
        data: pc.localDescription,
      });
    } else if (message.signalType === 'answer') {
      if (pc) {
        await pc.setRemoteDescription(message.data as RTCSessionDescriptionInit);
      }
    } else if (message.signalType === 'ice-candidate') {
      if (pc) {
        await pc.addIceCandidate(message.data as RTCIceCandidateInit);
      }
    }
  }


  // === P2P Messaging ===

  private handleP2PMessage(message: Record<string, unknown>, fromId: string): void {
    const peer = this._peers().get(fromId);
    const fromName = peer?.name || 'Unknown';

    switch (message['type']) {
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
              fromName,
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
              fromName,
            },
          })
        );
        break;
    }
  }

  sendToPeer(peerId: string, message: object): boolean {
    const dc = this.dataChannels.get(peerId);
    if (dc?.readyState === 'open') {
      dc.send(JSON.stringify(message));
      return true;
    }
    return false;
  }

  broadcastToPeers(message: object): void {
    const data = JSON.stringify(message);
    this.dataChannels.forEach((dc) => {
      if (dc.readyState === 'open') {
        dc.send(data);
      }
    });
  }

  // === Inference Distribution ===

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

  sendInferenceStream(taskId: string, targetId: string, chunk: string): void {
    this.sendToPeer(targetId, {
      type: 'inference-stream',
      taskId,
      chunk,
    });
  }

  // === Direct Message via Server (fallback) ===

  private handleDirectMessage(message: {
    data: unknown;
    fromId: string;
    fromName: string;
  }): void {
    const data = message.data as Record<string, unknown>;
    this.handleP2PMessage(data, message.fromId);
  }

  sendDirectViaServer(targetId: string, data: object): void {
    this.sendToServer({
      type: 'direct',
      targetId,
      data,
    });
  }
}
