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

  readonly roomCode = this.p2pSync.roomCode;
  readonly connectionMode = this.p2pSync.connectionMode;
  readonly peers = this.p2pSync.peers;
  readonly connectedPeers = this.p2pSync.connectedPeers;
  readonly isConnected = this.p2pSync.isConnected;
  readonly error = this.p2pSync.error;

  readonly deviceInfo = signal(this.p2pSync.getDeviceInfo());
  readonly joinCode = signal('');
  readonly isLoading = signal(false);
  readonly editingName = signal(false);
  readonly newDeviceName = signal('');

  async createRoom(): Promise<void> {
    this.isLoading.set(true);
    try {
      await this.p2pSync.createRoom();
    } catch (e) {
      console.error('Error creating room:', e);
    } finally {
      this.isLoading.set(false);
    }
  }

  async joinRoom(): Promise<void> {
    const code = this.joinCode().trim().toUpperCase();
    if (code.length < 4) return;

    this.isLoading.set(true);
    try {
      await this.p2pSync.joinRoom(code);
      this.joinCode.set('');
    } catch (e) {
      console.error('Error joining room:', e);
    } finally {
      this.isLoading.set(false);
    }
  }

  disconnect(): void {
    this.p2pSync.disconnect();
  }

  copyCode(): void {
    const code = this.roomCode();
    if (code) {
      navigator.clipboard.writeText(code);
    }
  }

  startEditName(): void {
    this.editingName.set(true);
    this.newDeviceName.set(this.deviceInfo().name);
  }

  saveName(): void {
    const name = this.newDeviceName().trim();
    if (name) {
      this.p2pSync.setDeviceName(name);
      this.deviceInfo.set({ ...this.deviceInfo(), name });
    }
    this.editingName.set(false);
  }

  getStatusColor(status: PeerDevice['status']): string {
    const colors: Record<PeerDevice['status'], string> = {
      connected: 'bg-green-500',
      connecting: 'bg-yellow-500',
      error: 'bg-red-500',
    };
    return colors[status] || 'bg-gray-500';
  }
}
