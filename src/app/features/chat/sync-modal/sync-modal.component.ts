import { Component, inject, signal, output } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { P2PSyncService, PeerDevice } from '../../../core/services/p2p-sync.service';

@Component({
  selector: 'app-sync-modal',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './sync-modal.component.html',
})
export class SyncModalComponent {
  private readonly p2pSync = inject(P2PSyncService);

  readonly close = output<void>();

  readonly peers = this.p2pSync.peers;
  readonly connectedPeers = this.p2pSync.connectedPeers;
  readonly syncToken = this.p2pSync.syncToken;
  readonly isConnected = this.p2pSync.isConnected;
  readonly connectionMode = this.p2pSync.connectionMode;
  readonly serverUrl = this.p2pSync.serverUrl;
  readonly error = this.p2pSync.error;

  readonly deviceInfo = this.p2pSync.getDeviceInfo();
  readonly editingName = signal(false);
  readonly newDeviceName = signal(this.deviceInfo.name);
  readonly editingServer = signal(false);
  readonly newServerUrl = signal(this.p2pSync.serverUrl());
  readonly joinToken = signal('');
  readonly activeTab = signal<'connect' | 'peers'>('connect');
  readonly isConnecting = signal(false);

  async connectToServer(): Promise<void> {
    this.isConnecting.set(true);
    try {
      await this.p2pSync.connectToServer();
    } catch (e) {
      console.error('Connection error:', e);
    } finally {
      this.isConnecting.set(false);
    }
  }

  disconnect(): void {
    this.p2pSync.disconnectFromServer();
  }

  async createRoom(): Promise<void> {
    const token = this.p2pSync.generateSyncToken();
    await this.p2pSync.joinRoom(token);
  }

  async joinRoom(): Promise<void> {
    const token = this.joinToken().trim().toUpperCase();
    if (token.length >= 4) {
      await this.p2pSync.joinRoom(token);
      this.joinToken.set('');
      this.activeTab.set('peers');
    }
  }

  leaveRoom(): void {
    this.p2pSync.leaveRoom();
  }

  startEditName(): void {
    this.editingName.set(true);
    this.newDeviceName.set(this.deviceInfo.name);
  }

  saveName(): void {
    const name = this.newDeviceName().trim();
    if (name) {
      this.p2pSync.setDeviceName(name);
      this.deviceInfo.name = name;
    }
    this.editingName.set(false);
  }

  startEditServer(): void {
    this.editingServer.set(true);
    this.newServerUrl.set(this.p2pSync.serverUrl());
  }

  saveServer(): void {
    const url = this.newServerUrl().trim();
    if (url) {
      this.p2pSync.setServerUrl(url);
    }
    this.editingServer.set(false);
  }

  copyToken(): void {
    const token = this.syncToken();
    if (token) {
      navigator.clipboard.writeText(token);
    }
  }

  getStatusColor(status: PeerDevice['status']): string {
    const colors: Record<PeerDevice['status'], string> = {
      connected: 'bg-green-500',
      connecting: 'bg-yellow-500',
      pending: 'bg-yellow-500',
      error: 'bg-red-500',
      discovered: 'bg-gray-500',
    };
    return colors[status] || 'bg-gray-500';
  }

  getStatusText(status: PeerDevice['status']): string {
    const texts: Record<PeerDevice['status'], string> = {
      connected: 'Conectado',
      connecting: 'Conectando...',
      pending: 'Pendiente',
      error: 'Error',
      discovered: 'Descubierto',
    };
    return texts[status] || 'Desconocido';
  }
}
