import { Component, input } from '@angular/core';

@Component({
  selector: 'app-card',
  standalone: true,
  imports: [],
  template: `
    <div class="rounded-3xl border border-slate-800 bg-slate-900/50 backdrop-blur-sm shadow-medium transition-all duration-200 hover:border-slate-600/50 hover:shadow-lg hover:-translate-y-1"
      [class]="paddingClass()">
      <ng-content />
    </div>
  `,
})
export class CardComponent {
  readonly padding = input<'none' | 'sm' | 'md' | 'lg'>('md');

  paddingClass(): string {
    switch (this.padding()) {
      case 'none': return 'p-0';
      case 'sm': return 'p-4';
      case 'lg': return 'p-8';
      default: return 'p-6';
    }
  }
}
