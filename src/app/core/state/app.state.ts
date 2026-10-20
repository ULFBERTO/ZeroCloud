import { Injectable, signal } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class AppStateService {
  private readonly _mobileNavOpen = signal(false);

  readonly mobileNavOpen = this._mobileNavOpen.asReadonly();

  openMobileNav(): void {
    this._mobileNavOpen.set(true);
  }

  closeMobileNav(): void {
    this._mobileNavOpen.set(false);
  }

  toggleMobileNav(): void {
    this._mobileNavOpen.update((open) => !open);
  }
}
