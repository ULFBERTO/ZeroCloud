import { Component, signal, inject } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { AppStateService } from './core/state';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  protected readonly title = signal('ZeroCloud');

  constructor() {
    // Inyectar AppStateService para inicializar el efecto de tema al arrancar la app
    inject(AppStateService);
  }
}
