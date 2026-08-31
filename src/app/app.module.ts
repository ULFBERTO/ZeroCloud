<mat-toolbar color="primary" class="top-bar">
  <button mat-icon-button (click)="sidenav.toggle()" aria-label="Toggle navigation">
    <mat-icon>menu</mat-icon>
  </button>
  <span class="app-title">{{ title() }}</span>
  <span class="spacer"></span>
  <mat-slide-toggle (change)="toggleTheme()" [checked]="isDarkTheme()">Dark</mat-slide-toggle>
</mat-toolbar>