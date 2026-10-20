import { Component, computed, inject, output, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { P2PSyncService, PeerDevice } from '../../../core/services/p2p-sync.service';
import { copyText } from '../../../core/utils/clipboard';
import { IconComponent } from '../../../shared/components';

@Component({
  selector: 'app-sync-modal',
  standalone: true,
  imports: [FormsModule, IconComponent],
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
  readonly myStats = this.p2pSync.myStats;
  readonly totalTasksCompleted = this.p2pSync.totalTasksCompleted;
  readonly totalTasksInProgress = this.p2pSync.totalTasksInProgress;

  readonly deviceInfo = signal(this.p2pSync.getDeviceInfo());
  readonly joinCode = signal('');
  readonly isLoading = signal(false);
  readonly editingName = signal(false);
  readonly newDeviceName = signal('');
  readonly showDisconnectConfirm = signal(false);
  readonly copied = signal(false);

  readonly isWorker = computed(() => this.connectionMode() === 'connected');
  readonly isHost = computed(() => this.connectionMode() === 'hosting');

  async createRoom(): Promise<void> {
    this.isLoading.set(true);
    try {
      await this.p2pSync.createRoom();
    } catch (error) {
      console.error('Error creating room:', error);
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
    } catch (error) {
      console.error('Error joining room:', error);
    } finally {
      this.isLoading.set(false);
    }
  }

  requestDisconnect(): void {
    this.showDisconnectConfirm.set(true);
  }

  confirmDisconnect(): void {
    this.p2pSync.disconnect();
    this.showDisconnectConfirm.set(false);
    this.close.emit();
  }

  cancelDisconnect(): void {
    this.showDisconnectConfirm.set(false);
  }

  disconnect(): void {
    this.p2pSync.disconnect();
    this.close.emit();
  }

  async copyCode(): Promise<void> {
    const code = this.roomCode();
    if (!code) return;
    if (await copyText(code)) {
      this.copied.set(true);
      window.setTimeout(() => this.copied.set(false), 1600);
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

  statusClass(status: PeerDevice['status']): string {
    const colors: Record<PeerDevice['status'], string> = {
      connected: 'bg-positive',
      connecting: 'bg-warning',
      error: 'bg-negative',
    };
    return `h-1.5 w-1.5 shrink-0 rounded-full ${colors[status] ?? 'bg-faint'}`;
  }
}
