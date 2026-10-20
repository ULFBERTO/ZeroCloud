import { Component, input, output } from '@angular/core';

@Component({
  selector: 'app-dialog',
  standalone: true,
  imports: [],
  template: `
    @if (open()) {
      <div class="fixed inset-0 z-50 flex items-center justify-center">
        <div class="fixed inset-0 bg-slate-950/70 backdrop-blur-sm transition-opacity" (click)="close.emit()"></div>
        <div class="relative w-full max-w-lg p-4 mx-auto transform transition-all duration-200"
          [class.scale-95]="!open()"
          [class.scale-100]="open()"
          (click)="$event.stopPropagation()">
          <div class="glass-panel rounded-3xl p-6">
            <div class="flex items-start justify-between mb-4">
              <h3 class="text-lg font-semibold text-slate-100">{{ title() }}</h3>
              <button (click)="close.emit()" class="p-1 text-slate-400 hover:text-slate-200 rounded-lg transition-colors">✕</button>
            </div>
            <div class="text-slate-400 text-sm mb-6 leading-relaxed">
              <ng-content />
            </div>
            <div class="flex justify-end gap-2">
              <ng-content select="[footer]"></ng-content>
            </div>
          </div>
        </div>
      </div>
    }
  `,
})
export class DialogComponent {
  readonly open = input<boolean>(false);
  readonly title = input<string>('');
  readonly close = output<void>();
}
